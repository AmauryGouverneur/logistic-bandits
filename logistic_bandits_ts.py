# logistic_bandits_ts.py

import os
import math
from typing import Optional

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

from tqdm import trange
import numpy as np
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

def sample_prior_theta(d, device=device, dtype=def_dtype):
    x = torch.randn(d, device=device, dtype=dtype)
    return x / x.norm().clamp_min(1e-12)

# ------------------------ batched experiment runner ------------------------

@torch.no_grad()
def run_logistic_bandits_TS_exp(
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
    append: bool = False,   # append to existing *.pt if present
):
    """
    Run `num_exp` independent experiments in chunks of size `batch_size`,
    with Metroplois-Hasting posterior sampling 

    Saves per-run to:
      results_experiments/logistic_ts_all_beta_{beta}_d_{d}.pt   (N, T)
    and mean to:
      results_experiments/logistic_ts_avg_beta_{beta}_d_{d}.pt   (T,)

    If append=True and the per-run file exists, appends `num_exp` new runs.
    """
    os.makedirs(save_dir, exist_ok=True)
    if seed is not None:
        set_seed(seed)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    use_cuda = (dev.type == 'cuda')
    K = chains if chains is not None else (256 if use_cuda else 32)

    file_avg = os.path.join(save_dir, f"logistic_ts_avg_beta_{beta}_d_{d}.pt")
    file_all = os.path.join(save_dir, f"logistic_ts_all_beta_{beta}_d_{d}.pt")

    # If not appending and file exists, just load & return
    if os.path.exists(file_all) and not append:
        print(f"Loading existing batched results from {file_all}…")
        all_runs = torch.load(file_all, map_location=dev)
        if all_runs.shape[1] != T:
            raise ValueError(f"Existing runs have T={all_runs.shape[1]} but T={T} was requested.")
        avg = all_runs.mean(dim=0)
        return avg, all_runs

    # We will generate 'num_exp' NEW runs into new_runs and then (optionally) append
    n_chunks = math.ceil(num_exp / batch_size)
    new_runs = torch.zeros(num_exp, T, device=dev, dtype=dtype)

    exp_idx = 0
    outer = trange(n_chunks, desc=f"MH-batched β={beta}, d={d}, B={batch_size}, K={K}", disable=not progress)
    for _ in outer:
        B = min(batch_size, num_exp - exp_idx)

        sampler = MHSphereSampler(d=d, kappa=kappa, device=dev, dtype=dtype)

        # (B,d) different theta_star for each experiment
        theta_star = torch.randn(B, d, device=dev, dtype=dtype)
        theta_star = theta_star / theta_star.norm(dim=1, keepdim=True).clamp_min(1e-12)

        # Histories (grow along time dimension)
        A_bt_d = torch.empty(B, 0, d, device=dev, dtype=dtype)   # (B, t, d)
        r_bt   = torch.empty(B, 0, device=dev, dtype=dtype)      # (B, t)

        # Warm start (one interaction)
        a0 = torch.randn(B, d, device=dev, dtype=dtype)
        a0 = a0 / a0.norm(dim=1, keepdim=True).clamp_min(1e-12)
        z0 = beta * (a0 * theta_star).sum(dim=1)
        r0 = torch.bernoulli(torch.sigmoid(z0))
        A_bt_d = torch.cat([A_bt_d, a0.unsqueeze(1)], dim=1)        # (B,1,d)
        r_bt   = torch.cat([r_bt, r0.unsqueeze(1)], dim=1)          # (B,1)

        # Persistent chains per experiment: (B, K, d)
        Theta_bkd = torch.randn(B, K, d, device=dev, dtype=dtype)
        Theta_bkd = Theta_bkd / Theta_bkd.norm(dim=2, keepdim=True).clamp_min(1e-12)

        regrets = torch.zeros(B, T, device=dev, dtype=dtype)
        opt_reward = torch.sigmoid(torch.tensor(beta, device=dev, dtype=dtype))  # phi(1)

        for t in range(T):
            # log posterior for all B experiments & K chains
            def logp_group(Theta_bkd_local: torch.Tensor) -> torch.Tensor:
                # Z: (B, t, K) = beta * (A @ Theta^T)
                Z = beta * torch.bmm(A_bt_d, Theta_bkd_local.transpose(1, 2))
                return (r_bt.unsqueeze(2) * Z - torch.nn.functional.softplus(Z)).sum(dim=1)  # (B, K)

            # Do a few MH steps starting from previous Theta_bkd
            Theta_bkd, _ = sampler.mh_step(Theta_bkd, logp_group, n_steps=mh_steps)

            # Thompson action per experiment: pick a random chain sample
            idx = torch.randint(0, K, (B,), device=dev)
            a_t = Theta_bkd[torch.arange(B, device=dev), idx, :]   # (B, d)

            # Environment step (Bernoulli reward)
            z_t = beta * (a_t * theta_star).sum(dim=1)
            r_t = torch.bernoulli(torch.sigmoid(z_t))

            # Append to history
            A_bt_d = torch.cat([A_bt_d, a_t.unsqueeze(1)], dim=1)
            r_bt   = torch.cat([r_bt, r_t.unsqueeze(1)], dim=1)

            # Rao–Blackwellized expected reward under current posterior
            proj = (Theta_bkd * theta_star.unsqueeze(1)).sum(dim=2)   # (B, K)
            mean_reward_est = torch.sigmoid(beta * proj).mean(dim=1)  # (B,)
            regrets[:, t] = opt_reward - mean_reward_est

        new_runs[exp_idx:exp_idx + B] = regrets
        exp_idx += B

    # Combine with previous if appending
    if append and os.path.exists(file_all):
        prev = torch.load(file_all, map_location="cpu")
        if prev.shape[1] != T:
            raise ValueError(f"Existing runs have T={prev.shape[1]} but T={T} was requested for append.")
        all_runs = torch.cat([prev.to(dev), new_runs], dim=0)
    else:
        all_runs = new_runs

    # Save & return
    avg = all_runs.mean(dim=0)
    torch.save(avg.detach().cpu(), file_avg)
    torch.save(all_runs.detach().cpu(), file_all)
    print(f"Saved batched average to {file_avg}\nSaved batched per-run to {file_all} (N={all_runs.shape[0]})")
    return avg, all_runs


# ------------------------ runs experiments on multiple betas -------------------------
def sweep_betas(
    betas=None,
    d: int = 10,
    T: int = 200,
    num_exp: int = 120,
    batch_size: int = 12,
    chains: int = 192,
    mh_steps: int = 10,
    progress: bool = True,
    save_dir: str = "results_experiments",
    append: bool = False,
):
    """
    Run logistic TS experiments for a list of betas.

    Saves one file per beta under `save_dir`.
    """
    if betas is None:
        betas = np.r_[0.25:4.0+0.25:0.25,  4.5:10.0+0.5:0.5].tolist()

    results = {}
    for beta in betas:
        avg, runs = run_logistic_bandits_TS_exp(
            d=d, beta=float(beta), T=T,
            num_exp=num_exp, batch_size=batch_size,
            chains=chains, mh_steps=mh_steps,
            progress=progress, save_dir=save_dir, append=append
        )
        print(f"β={beta}: saved per-run tensor {runs.shape}")
        results[beta] = (avg, runs)
    return results


if __name__ == "__main__":
    sweep_betas()