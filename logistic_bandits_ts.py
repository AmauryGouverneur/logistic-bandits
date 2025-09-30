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

# ------------------------ experiment runner ------------------------

@torch.no_grad()
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
):
    """
    Run `num_exp` independent experiments in chunks of size `batch_size`,
    using Metropolis–Hastings posterior sampling.

    Saves per-run to:
      results_experiments/logistic_ts_all_beta_{beta}_d_{d}.pt   (num_exp, T)
    and mean to:
      results_experiments/logistic_ts_avg_beta_{beta}_d_{d}.pt   (T,)

    If append=True and the per-run file exists, appends `num_exp` new runs.
    """
    os.makedirs(save_dir, exist_ok=True)
    if seed is not None:
        torch.manual_seed(seed)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    use_cuda = dev.type == "cuda"
    N = chains if chains is not None else (256 if use_cuda else 32)

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

    # Prepare storage
    n_chunks = math.ceil(num_exp / batch_size)
    new_runs = torch.zeros(num_exp, T, device=dev, dtype=dtype)

    exp_idx = 0
    outer = trange(n_chunks, desc=f"MH-batched β={beta}, d={d}, B={batch_size}, N={N}", disable=not progress)
    for _ in outer:
        B = min(batch_size, num_exp - exp_idx)

        sampler = MHSphereSampler(d=d, kappa=kappa, device=dev, dtype=dtype)

        # (B,d) different theta_star for each experiment
        theta_star = torch.randn(B, d, device=dev, dtype=dtype)
        theta_star = theta_star / theta_star.norm(dim=1, keepdim=True).clamp_min(1e-12)

        # Histories
        A_bt_d = torch.empty(B, 0, d, device=dev, dtype=dtype)   # (B, t, d)
        r_bt   = torch.empty(B, 0, device=dev, dtype=dtype)      # (B, t)

        # Warm start with one interaction
        a0 = torch.randn(B, d, device=dev, dtype=dtype)
        a0 = a0 / a0.norm(dim=1, keepdim=True).clamp_min(1e-12)
        z0 = beta * (a0 * theta_star).sum(dim=1)
        r0 = torch.bernoulli(torch.sigmoid(z0))
        A_bt_d = torch.cat([A_bt_d, a0.unsqueeze(1)], dim=1)        # (B,1,d)
        r_bt   = torch.cat([r_bt, r0.unsqueeze(1)], dim=1)          # (B,1)

        # N chains per experiment: (B, N, d)
        Theta_bnd = torch.randn(B, N, d, device=dev, dtype=dtype)
        Theta_bnd = Theta_bnd / Theta_bnd.norm(dim=2, keepdim=True).clamp_min(1e-12)

        regrets = torch.zeros(B, T, device=dev, dtype=dtype)

        # Initial regret estimate (t=0) using MC
        dot_ijn = torch.einsum("bid,bjd->bij", Theta_bnd, Theta_bnd)  # (B,N,N)
        rew_cross = torch.sigmoid(beta * dot_ijn)
        rew_diag = rew_cross.diagonal(dim1=1, dim2=2)  # (B,N)
        regret_est = (rew_diag.sum(dim=1) / N) - (rew_cross.mean(dim=(1,2)))
        regrets[:, 0] = regret_est

        # Time loop
        for t in range(T-1):
            # log posterior for all B experiments & N chains
            def logp_group(Theta_bnd_local: torch.Tensor) -> torch.Tensor:
                Z = beta * torch.bmm(A_bt_d, Theta_bnd_local.transpose(1, 2))  # (B, t, N)
                return (r_bt.unsqueeze(2) * Z - nn.functional.softplus(Z)).sum(dim=1)  # (B, N)

            # MH update
            Theta_bnd, _ = sampler.mh_step(Theta_bnd, logp_group, n_steps=mh_steps)

            # --- Monte Carlo regret estimate ---
            dot_ijn = torch.einsum("bid,bjd->bij", Theta_bnd, Theta_bnd)  # (B,N,N)
            rew_cross = torch.sigmoid(beta * dot_ijn)
            rew_diag = rew_cross.diagonal(dim1=1, dim2=2)  # (B,N)
            regret_est = (rew_diag.sum(dim=1) / N) - (rew_cross.mean(dim=(1,2)))
            regrets[:, t+1] = regret_est

            # Thompson action per experiment: pick a random chain sample
            idx = torch.randint(0, N, (B,), device=dev)
            a_t = Theta_bnd[torch.arange(B, device=dev), idx, :]   # (B, d)

            # Environment step (Bernoulli reward)
            z_t = beta * (a_t * theta_star).sum(dim=1)
            r_t = torch.bernoulli(torch.sigmoid(z_t))

            # Append to history
            A_bt_d = torch.cat([A_bt_d, a_t.unsqueeze(1)], dim=1)
            r_bt   = torch.cat([r_bt, r_t.unsqueeze(1)], dim=1)

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
    num_exp: int = 1024,
    batch_size: int = 256,
    chains: int = 64,
    mh_steps: int = 16,
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