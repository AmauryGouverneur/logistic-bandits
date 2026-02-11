import os
import math
from typing import Optional

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
from torch import nn

from tqdm import trange
import numpy as np
from mh_sphere import MHSphereSampler 

# ------------------------------- utils -------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def_dtype = torch.float32


def phi(x, beta):
    return torch.sigmoid(beta * x)

def sample_prior_theta(d, device=device, dtype=def_dtype):
    x = torch.randn(d, device=device, dtype=dtype)
    return x / x.norm().clamp_min(1e-12)

def binary_entropy(p: torch.Tensor, eps: float = 1e-8):
    """
    h(p) = -p log p - (1-p) log(1-p)
    """
    p = p.clamp(eps, 1 - eps)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))



# --- drop-in replacement for run_logistic_bandits_TS_exp in logistic_bandits_ts.py ---

def run_logistic_bandits_TS_exp(
    d: int,
    beta: float,
    T: int = 200,
    num_exp: int = 1024,
    batch_size: int = 256,
    kappa: float = 8.0,
    mh_steps: int = 16,
    chains: Optional[int] = None,
    progress: bool = True,
    save_dir: str = "results_experiments",
    seed: Optional[int] = 0,
    append: bool = False,   # append to existing *.pt if present

    # NEW:
    policy: str = "TS",      # "TS" or "EVDS"
    evr_eps: float = 1e-8,   # stability for EVR denominator
    use_true_regret: bool = True,  # log true expected per-step regret vs theta_star
    save_post_regret: bool = False # optionally also save posterior-regret estimate
):
    """
    Run `num_exp` independent experiments in chunks of size `batch_size`,
    using Metropolis–Hastings posterior sampling on the sphere.

    Policies:
      - TS:    Thompson Sampling (sample theta ~ posterior, play a=theta)
      - EVDS:  Greedy EVR minimization over candidate actions = posterior samples

    Regret logging:
      - use_true_regret=True:  regret_t = sigmoid(beta) - sigmoid(beta <a_t, theta_star>)
      - use_true_regret=False: uses the existing posterior regret estimate (as in your original code)

    Saves per-run to:
      results_experiments/logistic_{policy}_all_beta_{beta}_d_{d}.pt   (num_exp, T)
    and mean to:
      results_experiments/logistic_{policy}_avg_beta_{beta}_d_{d}.pt   (T,)

    If save_post_regret=True, also saves *_post_all_* and *_post_avg_*.
    """
    os.makedirs(save_dir, exist_ok=True)
    if seed is not None:
        torch.manual_seed(seed)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    use_cuda = dev.type == "cuda"
    N = chains if chains is not None else (256 if use_cuda else 32)

    pol = policy.lower()
    file_avg = os.path.join(save_dir, f"logistic_{pol}_avg_beta_{beta}_d_{d}.pt")
    file_all = os.path.join(save_dir, f"logistic_{pol}_all_beta_{beta}_d_{d}.pt")

    file_avg_post = os.path.join(save_dir, f"logistic_{pol}_post_avg_beta_{beta}_d_{d}.pt")
    file_all_post = os.path.join(save_dir, f"logistic_{pol}_post_all_beta_{beta}_d_{d}.pt")

    # If not appending and file exists, just load & return
    if os.path.exists(file_all) and not append:
        print(f"Loading existing batched results from {file_all}…")
        all_runs = torch.load(file_all, map_location=dev)
        if all_runs.shape[1] != T:
            raise ValueError(f"Existing runs have T={all_runs.shape[1]} but T={T} was requested.")
        avg = all_runs.mean(dim=0)
        return avg, all_runs

    # Prepare storage
    n_chunks = math.ceil(num_exp / batch_size)
    new_runs = torch.zeros(num_exp, T, device=dev, dtype=dtype)

    new_runs_post = None
    if save_post_regret:
        new_runs_post = torch.zeros(num_exp, T, device=dev, dtype=dtype)

    exp_idx = 0
    outer = trange(n_chunks, desc=f"{policy}-MH β={beta}, d={d}, B={batch_size}, N={N}", disable=not progress)

    s_beta = torch.sigmoid(torch.tensor(beta, device=dev, dtype=dtype))

    for _ in outer:
        B = min(batch_size, num_exp - exp_idx)

        sampler = MHSphereSampler(d=d, kappa=kappa, device=dev, dtype=dtype)

        # (B,d) different theta_star for each experiment (unit sphere)
        theta_star = torch.randn(B, d, device=dev, dtype=dtype)
        theta_star = theta_star / theta_star.norm(dim=1, keepdim=True).clamp_min(1e-12)

        # Histories
        A_bt_d = torch.empty(B, 0, d, device=dev, dtype=dtype)   # (B, t, d)
        r_bt   = torch.empty(B, 0, device=dev, dtype=dtype)      # (B, t)

        # Warm start with one interaction (t=1)
        a0 = torch.randn(B, d, device=dev, dtype=dtype)
        a0 = a0 / a0.norm(dim=1, keepdim=True).clamp_min(1e-12)
        z0 = beta * (a0 * theta_star).sum(dim=1)
        r0 = torch.bernoulli(torch.sigmoid(z0))
        A_bt_d = torch.cat([A_bt_d, a0.unsqueeze(1)], dim=1)        # (B,1,d)
        r_bt   = torch.cat([r_bt, r0.unsqueeze(1)], dim=1)          # (B,1)

        # N chains per experiment: (B, N, d)
        Theta_bnd = torch.randn(B, N, d, device=dev, dtype=dtype)
        Theta_bnd = Theta_bnd / Theta_bnd.norm(dim=2, keepdim=True).clamp_min(1e-12)

        # =====================================================
        # GLM-UCB STATE (Frequentist)
        # =====================================================
        if policy.upper() == "GLM_UCB":

            lam = 1.0
            V = lam * torch.eye(d, device=dev).unsqueeze(0).repeat(B, 1, 1)

            theta_hat = torch.zeros(B, d, device=dev)



        regrets = torch.zeros(B, T, device=dev, dtype=dtype)
        regrets_post = torch.zeros(B, T, device=dev, dtype=dtype) if save_post_regret else None

        # --- log t=1 regret ---
        if use_true_regret:
            # expected regret vs true theta_star
            rew0 = torch.sigmoid(beta * (a0 * theta_star).sum(dim=1))
            regrets[:, 0] = s_beta - rew0
        else:
            # keep your original posterior-regret estimate at t=1
            dot_ijn = torch.einsum("bid,bjd->bij", Theta_bnd, Theta_bnd)  # (B,N,N)
            rew_cross = torch.sigmoid(beta * dot_ijn)
            rew_diag = rew_cross.diagonal(dim1=1, dim2=2)  # (B,N)
            regret_est = (rew_diag.sum(dim=1) / N) - (rew_cross.mean(dim=(1,2)))
            regrets[:, 0] = regret_est

        if save_post_regret:
            dot_ijn = torch.einsum("bid,bjd->bij", Theta_bnd, Theta_bnd)  # (B,N,N)
            rew_cross = torch.sigmoid(beta * dot_ijn)
            rew_diag = rew_cross.diagonal(dim1=1, dim2=2)  # (B,N)
            regret_est = (rew_diag.sum(dim=1) / N) - (rew_cross.mean(dim=(1,2)))
            regrets_post[:, 0] = regret_est

        
        # =====================================================
        # Warm-start GLM-UCB update
        # =====================================================
        if policy.upper() == "GLM_UCB":

            V = V + torch.einsum("bd,be->bde", a0, a0)

            for _ in range(3):

                z = beta * torch.einsum("btd,bd->bt", A_bt_d, theta_hat)
                p = torch.sigmoid(z)

                grad = torch.einsum("bt,btd->bd", (r_bt - p), A_bt_d)

                W = p * (1 - p)
                H = torch.einsum("bt,btd,bte->bde", W, A_bt_d, A_bt_d)

                H_reg = H + lam * torch.eye(d, device=dev)

                theta_hat = theta_hat + torch.linalg.solve(H_reg, grad)

        # Time loop for t = 2..T
        for t in range(T - 1):
            # log posterior for all B experiments & N chains
            def logp_group(Theta_bnd_local: torch.Tensor) -> torch.Tensor:
                Z = beta * torch.bmm(A_bt_d, Theta_bnd_local.transpose(1, 2))  # (B, t_hist, N)
                return (r_bt.unsqueeze(2) * Z - nn.functional.softplus(Z)).sum(dim=1)  # (B, N)

            # MH update
            Theta_bnd, _ = sampler.mh_step(Theta_bnd, logp_group, n_steps=mh_steps)

            # Optional posterior regret estimate each step (for diagnostics)
            if save_post_regret or (not use_true_regret):
                dot_ijn = torch.einsum("bid,bjd->bij", Theta_bnd, Theta_bnd)  # (B,N,N)
                rew_cross = torch.sigmoid(beta * dot_ijn)
                rew_diag = rew_cross.diagonal(dim1=1, dim2=2)  # (B,N)
                regret_est = (rew_diag.sum(dim=1) / N) - (rew_cross.mean(dim=(1,2)))
                if save_post_regret:
                    regrets_post[:, t + 1] = regret_est
                if not use_true_regret:
                    regrets[:, t + 1] = regret_est

            # ---------- choose action ----------
            if policy.upper() == "TS":
                idx = torch.randint(0, N, (B,), device=dev)
                a_t = Theta_bnd[torch.arange(B, device=dev), idx, :]   # (B, d)

            elif policy.upper() == "EVDS":
                # Candidates are the posterior samples themselves: a in {Theta_bnd[:, i, :]}
                # Compute reward matrix for all candidate i against posterior samples j:
                dot_ijn = torch.einsum("bid,bjd->bij", Theta_bnd, Theta_bnd)  # (B,N,N)
                rew_cross = torch.sigmoid(beta * dot_ijn)                     # (B,N,N)

                # For each candidate i, mean/var over j
                m_i = rew_cross.mean(dim=2)                                   # (B,N)
                v_i = rew_cross.var(dim=2, unbiased=False)                    # (B,N)

                # Delta(a) = sigmoid(beta) - E_theta sigmoid(beta <a,theta>)
                delta_i = s_beta - m_i                                        # (B,N)

                # EVR score (stabilized)
                gamma_i = (delta_i * delta_i) / (v_i + evr_eps)               # (B,N)

                i_min = torch.argmin(gamma_i, dim=1)                          # (B,)
                a_t = Theta_bnd[torch.arange(B, device=dev), i_min, :]        # (B,d)

            elif policy.upper() == "EVDS_GD":

                # --- Step 1: discrete EVDS score on posterior samples ---

                dot_ijn = torch.einsum("bid,bjd->bij", Theta_bnd, Theta_bnd)
                mu_ijn = torch.sigmoid(beta * dot_ijn)

                m_i = mu_ijn.mean(dim=2)
                v_i = mu_ijn.var(dim=2, unbiased=False)

                delta_i = s_beta - m_i
                score_i = (delta_i**2) / (v_i + evr_eps)

                # pick top M candidates
                M = 5
                top_idx = torch.topk(score_i, M, largest=False).indices   # (B,M)

                # --- Step 2: refine each candidate with GD ---

                best_score = torch.full((B,), float("inf"), device=dev)
                best_action = None

                for j in range(M):

                    # initialize from posterior candidate
                    a = Theta_bnd[torch.arange(B), top_idx[:, j]].clone()
                    a.requires_grad_(True)

                    optimizer = torch.optim.Adam([a], lr=0.05)

                    for _ in range(15):

                        optimizer.zero_grad()

                        dot = torch.einsum("bd,bnd->bn", a, Theta_bnd)
                        mu = torch.sigmoid(beta * dot)

                        m = mu.mean(dim=1)
                        v = mu.var(dim=1, unbiased=False)

                        delta = s_beta - m
                        loss = ((delta**2) / (v + evr_eps)).mean()

                        loss.backward()
                        optimizer.step()

                        # project back to sphere
                        with torch.no_grad():
                            a /= a.norm(dim=1, keepdim=True)

                    # evaluate refined score
                    with torch.no_grad():
                        dot = torch.einsum("bd,bnd->bn", a, Theta_bnd)
                        mu = torch.sigmoid(beta * dot)

                        m = mu.mean(dim=1)
                        v = mu.var(dim=1, unbiased=False)
                        delta = s_beta - m

                        refined_score = (delta**2) / (v + evr_eps)

                        improved = refined_score < best_score
                        best_score[improved] = refined_score[improved]

                        if best_action is None:
                            best_action = a.detach()
                        else:
                            best_action[improved] = a.detach()[improved]

                a_t = best_action




            elif policy.upper() == "IDS":
                # Candidates are posterior samples themselves

                # Reward matrix μ_ij = σ(β <θ_i, θ_j>)
                dot_ijn = torch.einsum("bid,bjd->bij", Theta_bnd, Theta_bnd)
                mu_ijn = torch.sigmoid(beta * dot_ijn)   # (B,N,N)

                # Posterior mean reward for each candidate i
                mu_bar_i = mu_ijn.mean(dim=2)            # (B,N)

                # Gap Δ_i = σ(β) - E μ(a,Θ)
                delta_i = s_beta - mu_bar_i              # (B,N)

                # Information gain:
                # I(a) = h(E μ) - E[h(μ)]
                H_bar = binary_entropy(mu_bar_i)         # (B,N)
                H_cond = binary_entropy(mu_ijn).mean(dim=2)

                info_i = (H_bar - H_cond).clamp_min(1e-8)

                # IDS score
                score_i = (delta_i ** 2) / info_i        # (B,N)

                # Pick minimizer
                i_min = torch.argmin(score_i, dim=1)

                a_t = Theta_bnd[torch.arange(B, device=dev), i_min, :]

            elif policy.upper() == "IDS_GD":

                idx0 = torch.randint(0, N, (B,), device=dev)
                a0 = Theta_bnd[torch.arange(B, device=dev), idx0, :]
                a = torch.nn.Parameter(a0.clone())

                optimizer = torch.optim.Adam([a], lr=0.05)

                for _ in range(20):
                    optimizer.zero_grad()

                    dot = torch.einsum("bd,bnd->bn", a, Theta_bnd)
                    mu = torch.sigmoid(beta * dot)

                    m = mu.mean(dim=1)
                    delta = s_beta - m

                    H_bar = binary_entropy(m)
                    H_cond = binary_entropy(mu).mean(dim=1)

                    info = (H_bar - H_cond).clamp_min(1e-8)

                    loss = ((delta ** 2) / info).mean()

                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        a.data /= a.data.norm(dim=1, keepdim=True)

                a_t = a.detach()

            elif policy.upper() == "IDS_2P":

                # Reward matrix μ_ij
                dot_ijn = torch.einsum("bid,bjd->bij", Theta_bnd, Theta_bnd)
                mu_ijn = torch.sigmoid(beta * dot_ijn)

                # Mean reward per candidate
                mu_bar = mu_ijn.mean(dim=2)          # (B,N)

                # Gap Δ_i
                delta = s_beta - mu_bar              # (B,N)

                # Info gain I_i
                H_bar = binary_entropy(mu_bar)
                H_cond = binary_entropy(mu_ijn).mean(dim=2)

                info = (H_bar - H_cond).clamp_min(1e-8)

                # Single-point IDS score
                score = delta**2 / info

                # Take top M candidates
                M = 10
                top = torch.topk(score, M, largest=False).indices   # (B,M)

                # Search best pair
                best_val = torch.full((B,), float("inf"), device=dev)
                best_i = None
                best_k = None
                best_p = None

                for u in range(M):
                    for v in range(u+1, M):

                        i = top[:, u]
                        k = top[:, v]

                        Δi, Δk = delta[torch.arange(B), i], delta[torch.arange(B), k]
                        Ii, Ik = info[torch.arange(B), i], info[torch.arange(B), k]

                        # closed-form minimizer of fractional quadratic
                        p = torch.clamp(
                            (Δk*Ik - Δi*Ik) / ((Δi-Δk)*(Ii-Ik) + 1e-8),
                            0.0, 1.0
                        )

                        Δp = p*Δi + (1-p)*Δk
                        Ip = p*Ii + (1-p)*Ik

                        val = (Δp**2) / Ip

                        improved = val < best_val
                        best_val[improved] = val[improved]
                        best_i = i if best_i is None else torch.where(improved, i, best_i)
                        best_k = k if best_k is None else torch.where(improved, k, best_k)
                        best_p = p if best_p is None else torch.where(improved, p, best_p)

                # Sample action from mixture
                u = torch.rand(B, device=dev)
                choose_i = u < best_p

                a_t = torch.where(
                    choose_i.unsqueeze(1),
                    Theta_bnd[torch.arange(B), best_i],
                    Theta_bnd[torch.arange(B), best_k]
                )

            elif policy.upper() == "BAYESUCB":
                dot = torch.einsum("bid,bjd->bij", Theta_bnd, Theta_bnd)
                mu = torch.sigmoid(beta * dot)   # (B,N,N)

                q = 1.0 - 1.0/(t+2)
                quant = torch.quantile(mu, q, dim=2)

                i_max = torch.argmax(quant, dim=1)
                a_t = Theta_bnd[torch.arange(B), i_max]

            elif policy.upper() == "GLM_UCB":

                Vinv = torch.linalg.solve(V, torch.eye(d, device=dev))


                A_cand = Theta_bnd  # candidates on sphere

                pred = torch.sigmoid(
                    beta * torch.einsum("bnd,bd->bn", A_cand, theta_hat)
                )

                bonus = torch.sqrt(
                    torch.einsum("bnd,bde,bne->bn", A_cand, Vinv, A_cand)
                )

                alpha_t = 2.0 * math.sqrt(d * math.log(t + 2))

                score = pred + alpha_t * bonus

                i_max = torch.argmax(score, dim=1)
                a_t = Theta_bnd[torch.arange(B), i_max]

            else:
                raise ValueError(f"Unknown policy={policy}. Use 'TS', 'EVDS', 'EVDS_GD', 'IDS', or 'IDS_2P'.")

            # Environment step (Bernoulli reward)
            z_t = beta * (a_t * theta_star).sum(dim=1)
            r_t = torch.bernoulli(torch.sigmoid(z_t))

            # True expected per-step regret (if requested)
            if use_true_regret:
                rew_t = torch.sigmoid(z_t)  # expected reward under theta_star
                regrets[:, t + 1] = s_beta - rew_t

            # Append to history
            A_bt_d = torch.cat([A_bt_d, a_t.unsqueeze(1)], dim=1)
            r_bt   = torch.cat([r_bt, r_t.unsqueeze(1)], dim=1)

            # =====================================================
            # GLM-UCB update (after observing reward)
            # =====================================================
            if policy.upper() == "GLM_UCB":

                # Update design matrix
                V = V + torch.einsum("bd,be->bde", a_t, a_t)

                # Newton updates for logistic MLE
                for _ in range(3):

                    z = beta * torch.einsum("btd,bd->bt", A_bt_d, theta_hat)
                    p = torch.sigmoid(z)

                    grad = torch.einsum("bt,btd->bd", (r_bt - p), A_bt_d)

                    W = p * (1 - p)
                    H = torch.einsum("bt,btd,bte->bde", W, A_bt_d, A_bt_d)

                    H_reg = H + lam * torch.eye(d, device=dev)

                    theta_hat = theta_hat + torch.linalg.solve(H_reg, grad)


        new_runs[exp_idx:exp_idx + B] = regrets
        if save_post_regret:
            new_runs_post[exp_idx:exp_idx + B] = regrets_post
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
    print(f"Saved average to {file_avg}\nSaved per-run to {file_all} (N={all_runs.shape[0]})")

    if save_post_regret:
        if append and os.path.exists(file_all_post):
            prevp = torch.load(file_all_post, map_location="cpu")
            if prevp.shape[1] != T:
                raise ValueError(f"Existing post runs have T={prevp.shape[1]} but T={T} was requested for append.")
            all_runs_post = torch.cat([prevp.to(dev), new_runs_post], dim=0)
        else:
            all_runs_post = new_runs_post

        avg_post = all_runs_post.mean(dim=0)
        torch.save(avg_post.detach().cpu(), file_avg_post)
        torch.save(all_runs_post.detach().cpu(), file_all_post)
        print(f"Saved post-average to {file_avg_post}\nSaved post per-run to {file_all_post} (N={all_runs_post.shape[0]})")

    return avg, all_runs



# ------------------------ runs experiments on multiple betas -------------------------
def sweep_betas(
    betas=None,
    d: int = 10,
    T: int = 200,
    num_exp: int = 1024,
    batch_size: int = 256,
    chains: int = 64,
    mh_steps: int = 16,
    progress: bool = True,
    save_dir: str = "results_experiments",
    append: bool = False,
    policy: str = "TS",
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
            progress=progress, save_dir=save_dir, append=append, policy=policy
        )
        print(f"β={beta}: saved per-run tensor {runs.shape}")
        results[beta] = (avg, runs)
    return results


if __name__ == "__main__":
    sweep_betas(policy="BAYESUCB")