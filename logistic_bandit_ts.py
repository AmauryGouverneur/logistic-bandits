# logistic_ts_persistent.py
# Drop-in rewrite that (i) keeps MH chains alive across rounds, (ii) adapts kappa,
# and (iii) saves *each* experiment's regret trajectory so you can compute CIs.

import os
import math
import torch
torch.backends.cuda.matmul.allow_tf32 = True   # Ampere TF32
torch.set_float32_matmul_precision("high")     # let matmul use TF32
torch.backends.cudnn.benchmark = True          # autotune convs / activations

import matplotlib.pyplot as plt
from tqdm import trange
from mh_sphere import MHSphereSampler

# ------------------------------- utils -------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
def_dtype = torch.float32

def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def phi(x, beta):
    return torch.sigmoid(beta * x)


def sample_reward(A, Theta, beta):
    p = phi(torch.dot(A, Theta), beta)
    return torch.bernoulli(p)


def sample_prior_theta(d, device=device, dtype=def_dtype):
    x = torch.randn(d, device=device, dtype=dtype)
    return x / x.norm().clamp_min(1e-12)


# ------------------------ core posterior pieces ----------------------

def build_cache_tensors(history_actions, history_rewards, sampler):
    if history_actions:
        A = torch.stack(history_actions, dim=0).to(sampler.device, sampler.dtype)
        r = torch.stack(history_rewards, dim=0).to(sampler.device, sampler.dtype)
    else:
        A = torch.empty(0, sampler.d, device=sampler.device, dtype=sampler.dtype)
        r = torch.empty(0, device=sampler.device, dtype=sampler.dtype)
    return A, r


def make_logp_batch(A, r, beta):
    # returns a function: (K,d) -> (K,)
    if A.numel() == 0:
        def logp_batch_empty(Theta):
            return torch.zeros(Theta.shape[0], device=Theta.device, dtype=Theta.dtype)
        return logp_batch_empty

    def logp_batch(Theta):
        Z = beta * (A @ Theta.T)  # (t,K)
        return (r[:, None] * Z - torch.nn.functional.softplus(Z)).sum(dim=0)  # (K,)

    return logp_batch


# ----------------------- Thompson sampling step ----------------------

def thompson_sampled_action(
    history_actions,
    history_rewards,
    beta,
    sampler: MHSphereSampler,
    mh_steps=30,
    chains=None,
    return_batch=False,
    state=None,
    adapt_kappa=True,
    kappa_bounds=(1.0, 200.0),
):
    d = history_actions[0].numel() if history_actions else sampler.d

    # cache data for fast batched logp
    A, r = build_cache_tensors(history_actions, history_rewards, sampler)
    logp_batch = make_logp_batch(A, r, beta)

    use_cuda = sampler.device.type == 'cuda'
    K = chains if chains is not None else (256 if use_cuda else 32)

    # initialize or continue chains
    if state is not None and state.shape == (K, d):
        Theta0 = state
    else:
        mu0 = (history_actions[-1] if history_actions else sample_prior_theta(d, sampler.device, sampler.dtype))
        mu0 = mu0 / mu0.norm().clamp_min(1e-12)
        Theta0 = sampler._vmf_sample_batch(mu0.expand(K, -1))  # (K,d)

    ThetaT, logpT, accept_mask = sampler.step_batch(Theta0, logp_fn_batch=logp_batch, n_steps=mh_steps, return_accept_mask=True)

    # simple online adaptation of kappa to keep acceptance in [0.2, 0.4]
    if adapt_kappa and accept_mask is not None:
        acc_rate = accept_mask.float().mean().item()
        if acc_rate < 0.20:
            sampler.kappa = min(kappa_bounds[1], sampler.kappa * 1.15)
        elif acc_rate > 0.40:
            sampler.kappa = max(kappa_bounds[0], sampler.kappa * 0.87)

    idx = torch.randint(0, K, (1,), device=ThetaT.device).item()
    action = ThetaT[idx]

    new_state = ThetaT
    if return_batch:
        return action, ThetaT, new_state
    return action, new_state


# ------------------------ environment + regret -----------------------

def bandit_experiment(d, beta, T, sampler: MHSphereSampler, mh_steps=30, chains=None, rng_seed=None):
    if rng_seed is not None:
        set_seed(rng_seed)

    history_actions, history_rewards = [], []
    expected_regret = torch.zeros(T, device=sampler.device, dtype=sampler.dtype)

    theta_star = sample_prior_theta(d, device=sampler.device, dtype=sampler.dtype)

    # warm start with one interaction
    action0 = sample_prior_theta(d, device=sampler.device, dtype=sampler.dtype)
    reward0 = sample_reward(action0, theta_star, beta)
    history_actions.append(action0); history_rewards.append(reward0)

    opt_reward = phi(torch.tensor(1.0, device=sampler.device, dtype=sampler.dtype), beta)

    state = None
    for t in range(T):
        action, Theta_batch, state = thompson_sampled_action(
            history_actions, history_rewards, beta,
            sampler=sampler, mh_steps=mh_steps, chains=chains,
            return_batch=True, state=state
        )

        reward = sample_reward(action, theta_star, beta)
        history_actions.append(action); history_rewards.append(reward)

        # Rao–Blackwellized expected regret estimate: E[phi(beta * <theta*, theta_k>)] over posterior samples
        proj = Theta_batch @ theta_star  # (K,)
        mean_reward_est = phi(proj, beta).mean()
        expected_regret[t] = (opt_reward - mean_reward_est)

    return expected_regret  # shape (T,)


# ----------------------------- runner --------------------------------

def run_logistic_bandits_TS_exp(
    d,
    beta,
    T=200,
    num_exp=500,
    kappa=8.0,
    mh_steps=30,
    progress=False,
    chains=None,
    save_dir="results_experiments",
    seed=0,
    append=False,          # NEW: if True, append num_exp runs to existing file_all
):
    """
    Run logistic TS experiments and save:
      - per-experiment regrets to .../logistic_ts_all_beta_{beta}_d_{d}.pt  (shape: (N, T))
      - mean regret to .../logistic_ts_avg_beta_{beta}_d_{d}.pt              (shape: (T,))

    If append=True and the per-experiment file exists, the function appends `num_exp`
    new runs (with seed offset) to the existing data. T must match exactly when appending.
    """
    os.makedirs(save_dir, exist_ok=True)
    file_avg = os.path.join(save_dir, f"logistic_ts_avg_beta_{beta}_d_{d}.pt")
    file_all = os.path.join(save_dir, f"logistic_ts_all_beta_{beta}_d_{d}.pt")

    # Decide whether we are loading, appending, or starting fresh
    if os.path.exists(file_all) and not append:
        # Just load what exists and return (no new runs)
        print(f"Loading per-experiment results from {file_all}…")
        all_runs = torch.load(file_all, map_location=device)
        if all_runs.shape[1] != T:
            raise ValueError(
                f"Existing runs have T={all_runs.shape[1]} but T={T} was requested. "
                "Use the same T to reuse results."
            )
        avg = all_runs.mean(dim=0)
        return avg, all_runs  # (T,), (N, T)

    # Prepare sampler
    sampler = MHSphereSampler(d=d, kappa=kappa, device=device, dtype=def_dtype)

    # If appending, load existing and extend; otherwise create new tensor
    if append and os.path.exists(file_all):
        print(f"Appending {num_exp} runs to existing {file_all}…")
        prev_runs = torch.load(file_all, map_location=device)
        if prev_runs.shape[1] != T:
            raise ValueError(
                f"Existing runs have T={prev_runs.shape[1]} but T={T} was requested. "
                "Use the same T to append."
            )
        start_idx = prev_runs.shape[0]
        all_runs = torch.zeros(start_idx + num_exp, T, device=device, dtype=def_dtype)
        all_runs[:start_idx] = prev_runs.to(device=device, dtype=def_dtype)
    else:
        print(f"Running experiments for beta={beta}, d={d}, num_exp={num_exp}…")
        start_idx = 0
        all_runs = torch.zeros(num_exp, T, device=device, dtype=def_dtype)

    # Seed (offset when appending so runs are unique)
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    iterator = trange(num_exp, desc=f"β={beta}, d={d}", disable=not progress)
    for i in iterator:
        run_seed = None if seed is None else (seed + start_idx + i)
        all_runs[start_idx + i] = bandit_experiment(
            d, beta, T, sampler, mh_steps=mh_steps, chains=chains, rng_seed=run_seed
        )

    # Save & return
    avg = all_runs.mean(dim=0)
    torch.save(avg, file_avg)
    torch.save(all_runs, file_all)
    print(f"Saved average to {file_avg}\nSaved all runs to {file_all}")
    return avg, all_runs


# ======================= Batched parallel experiments (GPU) =======================
# Run B experiments in parallel, in chunks, with persistent MH per experiment.
# Requires: MHSphereSampler, phi, sample_prior_theta already defined above.

import math
from typing import Optional

def _batched_logp_group(A_bt_d, r_bt, Theta_bkd, beta):
    """
    A_bt_d:   (B, t, d)
    r_bt:     (B, t)
    Theta_bkd:(B, K, d)
    return:   (B, K) log-likelihood sums for each experiment and chain
    """
    # Z: (B, t, K) = beta * (A @ Theta^T)
    Z = beta * torch.bmm(A_bt_d, Theta_bkd.transpose(1, 2))
    # sum over t: r*Z - softplus(Z)
    # r_bt -> (B, t, 1)
    ll = (r_bt.unsqueeze(2) * Z - torch.nn.functional.softplus(Z)).sum(dim=1)
    return ll  # (B, K)

def _vmf_sample_batch_batched(sampler: MHSphereSampler, Theta_bkd):
    """
    Propose vMF RW for each of the B*K current states.
    Theta_bkd: (B, K, d) -> returns (B, K, d)
    """
    B, K, d = Theta_bkd.shape
    flat = Theta_bkd.reshape(B * K, d)
    flat_prop = sampler._vmf_sample_batch(flat)       # (B*K, d)
    return flat_prop.reshape(B, K, d)

from contextlib import nullcontext

def _mh_step_batched(sampler: MHSphereSampler, Theta_bkd, logp_fn_group, n_steps=12, use_amp=True):
    """
    Persistent MH over (B,K,d) states with optional CUDA autocast (bf16).
    Theta_bkd: (B,K,d); logp_fn_group: (B,K,d)->(B,K)
    Returns: (Theta_bkd_new, logp_bk)
    """
    device, dtype = sampler.device, sampler.dtype
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if (use_amp and device.type == "cuda") else nullcontext()

    with autocast_ctx:
        Theta = Theta_bkd.to(device, dtype)
        Theta = Theta / Theta.norm(dim=2, keepdim=True).clamp_min(1e-12)
        logp  = logp_fn_group(Theta)  # (B,K)

        for _ in range(n_steps):
            prop = _vmf_sample_batch_batched(sampler, Theta)   # (B,K,d)
            logp_prop = logp_fn_group(prop)                    # (B,K)
            u = torch.rand_like(logp).log()
            accept = u < (logp_prop - logp)
            if accept.any():
                Theta[accept] = prop[accept]
                logp[accept]  = logp_prop[accept]

    return Theta, logp

# (Optional) compile for extra speed (PyTorch 2.x)
try:
    _mh_step_batched = torch.compile(_mh_step_batched, fullgraph=False)
except Exception:
    pass

@torch.no_grad()
def run_logistic_bandits_TS_exp_batched(
    d: int,
    beta: float,
    T: int = 200,
    num_exp: int = 100,
    batch_size: int = 8,
    kappa: float = 8.0,
    mh_steps: int = 12,
    chains: Optional[int] = None,
    progress: bool = True,
    save_dir: str = "results_experiments",
    seed: Optional[int] = 0,
    append: bool = False,   # NEW: append to existing *_batched.pt if present
):
    """
    Run `num_exp` independent experiments in chunks of size `batch_size`,
    with exact MH posterior sampling, persistent chains, and vectorized likelihoods.

    Saves per-run to:
      results_experiments/logistic_ts_all_beta_{beta}_d_{d}_batched.pt   (N, T)
    and mean to:
      results_experiments/logistic_ts_avg_beta_{beta}_d_{d}_batched.pt   (T,)

    If append=True and the per-run file exists, appends `num_exp` new runs.
    """
    os.makedirs(save_dir, exist_ok=True)
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    use_cuda = (device.type == 'cuda')
    K = chains if chains is not None else (256 if use_cuda else 32)

    file_avg = os.path.join(save_dir, f"logistic_ts_avg_beta_{beta}_d_{d}_batched.pt")
    file_all = os.path.join(save_dir, f"logistic_ts_all_beta_{beta}_d_{d}_batched.pt")

    # If not appending and file exists, just load & return
    if os.path.exists(file_all) and not append:
        print(f"Loading existing batched results from {file_all}…")
        all_runs = torch.load(file_all, map_location=device)
        if all_runs.shape[1] != T:
            raise ValueError(f"Existing runs have T={all_runs.shape[1]} but T={T} was requested.")
        avg = all_runs.mean(dim=0)
        return avg, all_runs

    # We will generate 'num_exp' NEW runs into new_runs and then (optionally) append
    from tqdm import trange
    n_chunks = math.ceil(num_exp / batch_size)
    new_runs = torch.zeros(num_exp, T, device=device, dtype=dtype)

    exp_idx = 0
    outer = trange(n_chunks, desc=f"MH-batched β={beta}, d={d}, B={batch_size}, K={K}", disable=not progress)
    for _ in outer:
        B = min(batch_size, num_exp - exp_idx)

        sampler = MHSphereSampler(d=d, kappa=kappa, device=device, dtype=dtype)

        # (B,d) different theta_star
        theta_star = torch.randn(B, d, device=device, dtype=dtype)
        theta_star = theta_star / theta_star.norm(dim=1, keepdim=True).clamp_min(1e-12)

        # histories
        A_bt_d = torch.empty(B, 0, d, device=device, dtype=dtype)
        r_bt   = torch.empty(B, 0, device=device, dtype=dtype)

        # warm start
        a0 = torch.randn(B, d, device=device, dtype=dtype)
        a0 = a0 / a0.norm(dim=1, keepdim=True).clamp_min(1e-12)
        z0 = beta * (a0 * theta_star).sum(dim=1)
        r0 = torch.bernoulli(torch.sigmoid(z0))
        A_bt_d = torch.cat([A_bt_d, a0.unsqueeze(1)], dim=1)        # (B,1,d)
        r_bt   = torch.cat([r_bt, r0.unsqueeze(1)], dim=1)          # (B,1)

        # persistent chains (B,K,d)
        Theta_bkd = torch.randn(B, K, d, device=device, dtype=dtype)
        Theta_bkd = Theta_bkd / Theta_bkd.norm(dim=2, keepdim=True).clamp_min(1e-12)

        regrets = torch.zeros(B, T, device=device, dtype=dtype)
        opt_reward = torch.sigmoid(torch.tensor(beta, device=device, dtype=dtype))

        for t in range(T):
            def logp_group(Theta_bkd_local):
                Z = beta * torch.bmm(A_bt_d, Theta_bkd_local.transpose(1, 2))   # (B,t,K)
                return (r_bt.unsqueeze(2) * Z - torch.nn.functional.softplus(Z)).sum(dim=1)  # (B,K)

            Theta_bkd, _ = _mh_step_batched(sampler, Theta_bkd, logp_group, n_steps=mh_steps)

            # Thompson action per experiment
            idx = torch.randint(0, K, (B,), device=device)
            a_t = Theta_bkd[torch.arange(B, device=device), idx, :]   # (B,d)

            # env step
            z_t = beta * (a_t * theta_star).sum(dim=1)
            r_t = torch.bernoulli(torch.sigmoid(z_t))

            # grow history
            A_bt_d = torch.cat([A_bt_d, a_t.unsqueeze(1)], dim=1)
            r_bt   = torch.cat([r_bt, r_t.unsqueeze(1)], dim=1)

            # RB expected reward under posterior
            proj = (Theta_bkd * theta_star.unsqueeze(1)).sum(dim=2)   # (B,K)
            mean_reward_est = torch.sigmoid(beta * proj).mean(dim=1)
            regrets[:, t] = opt_reward - mean_reward_est

        new_runs[exp_idx:exp_idx + B] = regrets
        exp_idx += B

    # Combine with previous if appending
    if append and os.path.exists(file_all):
        prev = torch.load(file_all, map_location="cpu")
        if prev.shape[1] != T:
            raise ValueError(f"Existing runs have T={prev.shape[1]} but T={T} was requested for append.")
        all_runs = torch.cat([prev.to(device), new_runs], dim=0)
    else:
        all_runs = new_runs

    # Save & return
    avg = all_runs.mean(dim=0)
    torch.save(avg.detach().cpu(), file_avg)
    torch.save(all_runs.detach().cpu(), file_all)
    print(f"Saved batched average to {file_avg}\nSaved batched per-run to {file_all} (N={all_runs.shape[0]})")
    return avg, all_runs





# ----------------------------- plotting ------------------------------

def mean_and_ci(all_runs: torch.Tensor, alpha: float = 0.05):
    """
    all_runs: (num_exp, T)
    Returns: mean (T,), lower (T,), upper (T,) for a (1-alpha) CI using normal approx.
    """
    mean = all_runs.mean(dim=0)
    se = all_runs.std(dim=0, unbiased=True) / math.sqrt(all_runs.shape[0])
    z = 1.959963984540054  # ~ N(0,1) 97.5th percentile for 95% CI
    lower = mean - z * se
    upper = mean + z * se
    return mean, lower, upper


def demo_plot_with_ci(d=10, beta=2.0, T=200, num_exp=100, kappa=8.0, mh_steps=30, chains=None, progress=True):
    avg, all_runs = run_logistic_bandits_TS_exp(d, beta, T, num_exp, kappa, mh_steps, progress, chains)
    mean, lo, hi = mean_and_ci(all_runs)

    x = torch.arange(T)
    plt.figure()
    plt.plot(x.cpu().numpy(), mean.cpu().numpy(), label="TS (vMF–MH, persistent)")
    plt.fill_between(x.cpu().numpy(), lo.cpu().numpy(), hi.cpu().numpy(), alpha=0.25, label="95% CI")
    plt.xlabel("t")
    plt.ylabel("Expected regret")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    d, beta, T = 10, 2.0, 200
    avg, all_runs = run_logistic_bandits_TS_exp(d, beta, T=T, num_exp=100, kappa=8.0, mh_steps=30, progress=True)
    mean, lo, hi = mean_and_ci(all_runs)
    plt.figure()
    x = torch.arange(T)
    plt.plot(x.cpu().numpy(), mean.cpu().numpy(), label="TS (vMF–MH, persistent)")
    plt.fill_between(x.cpu().numpy(), lo.cpu().numpy(), hi.cpu().numpy(), alpha=0.25, label="95% CI")
    plt.xlabel("t"); plt.ylabel("Expected regret"); plt.legend(); plt.tight_layout(); plt.show()
