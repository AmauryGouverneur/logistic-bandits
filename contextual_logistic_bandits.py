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





# --- Contextual Logistic Bandits with Sphere Actions ---
# Feature map: phi(x,a)=x⊙a

@torch.no_grad()
def run_contextual_logistic_bandits_exp(
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
    append: bool = False,

    policy: str = "TS",
    evr_eps: float = 1e-8,
    use_true_regret: bool = True,
):
    """
    Contextual Logistic Bandit:

        r_t ~ Bernoulli( σ( β <θ*, x_t ⊙ a_t> ) )

    Actions live on the unit sphere.

    Policies supported:
      TS, EVDS, IDS, BayesUCB, GLM_UCB
    """

    os.makedirs(save_dir, exist_ok=True)
    if seed is not None:
        torch.manual_seed(seed)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    N = chains if chains is not None else 64

    pol = policy.lower()
    file_all = os.path.join(save_dir, f"contextual_{pol}_all_beta_{beta}_d_{d}.pt")
    file_avg = os.path.join(save_dir, f"contextual_{pol}_avg_beta_{beta}_d_{d}.pt")

    # Load existing
    if os.path.exists(file_all) and not append:
        print(f"Loading existing results from {file_all}")
        all_runs = torch.load(file_all, map_location=dev)
        return all_runs.mean(0), all_runs

    # Storage
    n_chunks = math.ceil(num_exp / batch_size)
    all_regrets = torch.zeros(num_exp, T, device=dev)

    exp_idx = 0
    outer = trange(n_chunks,
                   desc=f"{policy} Contextual β={beta}",
                   disable=not progress)

    s_beta = torch.sigmoid(torch.tensor(beta, device=dev))

    for _ in outer:

        B = min(batch_size, num_exp - exp_idx)

        # True parameter θ*
        theta_star = torch.randn(B, d, device=dev)
        theta_star = theta_star / theta_star.norm(dim=1, keepdim=True)

        # Posterior chains Θ
        Theta_bnd = torch.randn(B, N, d, device=dev)
        Theta_bnd = Theta_bnd / Theta_bnd.norm(dim=2, keepdim=True)

        sampler = MHSphereSampler(d=d, kappa=kappa, device=dev)

        # History: store φ(x,a)
        Phi_hist = torch.empty(B, 0, d, device=dev)
        r_hist   = torch.empty(B, 0, device=dev)

        regrets = torch.zeros(B, T, device=dev)

        # ------------------ main loop ------------------
        for t in range(T):

            # Sample context x_t
            x_t = torch.randn(B, d, device=dev)
            x_t = x_t / x_t.norm(dim=1, keepdim=True)

            # ---------- MH posterior update ----------
            if t > 0:

                def logp_group(Theta_local):

                    Z = beta * torch.bmm(
                        Phi_hist,
                        Theta_local.transpose(1, 2)
                    )  # (B,t,N)

                    return (r_hist.unsqueeze(2) * Z
                            - nn.functional.softplus(Z)).sum(dim=1)
                
                sampler.kappa = (
                1.0 + 1.0 * beta**0.8 * (t + 1)**0.4
                )

                Theta_bnd, _ = sampler.mh_step(
                    Theta_bnd, logp_group, n_steps=mh_steps, verbose=False) #(t % 20 == 0)


            # ---------- Candidate reward matrix ----------
            # μ_ij = σ(β <Θ_j, x⊙a_i>)
            dot = torch.einsum(
                "bid,bjd->bij",
                Theta_bnd,
                Theta_bnd
            )  # (B,N,N)

            # Context enters here:
            Theta_eff = Theta_bnd * x_t.unsqueeze(1)

            mu = torch.sigmoid(
                beta * torch.einsum("bid,bjd->bij",
                                    Theta_eff,
                                    Theta_bnd)
            )

            mu_bar = mu.mean(dim=2)
            delta = s_beta - mu_bar

            # ---------- Policy selection ----------
            if policy.upper() == "TS":
                idx = torch.randint(0, N, (B,), device=dev)
                a_t = Theta_bnd[torch.arange(B), idx]

            elif policy.upper() == "EVDS":
                v = mu.var(dim=2, unbiased=False)
                score = delta**2 / (v + evr_eps)
                i_min = torch.argmin(score, dim=1)
                a_t = Theta_bnd[torch.arange(B), i_min]

            elif policy.upper() == "IDS":
                H_bar = binary_entropy(mu_bar)
                H_cond = binary_entropy(mu).mean(dim=2)
                info = (H_bar - H_cond).clamp_min(1e-8)
                score = delta**2 / info
                i_min = torch.argmin(score, dim=1)
                a_t = Theta_bnd[torch.arange(B), i_min]

            elif policy.upper() == "BAYESUCB":
                q = 1.0 - 1.0/(t+2)
                quant = torch.quantile(mu, q, dim=2)
                i_max = torch.argmax(quant, dim=1)
                a_t = Theta_bnd[torch.arange(B), i_max]

            else:
                raise ValueError(f"Unknown policy {policy}")

            # ---------- Environment step ----------
            phi_t = x_t * a_t
            z_t = beta * (phi_t * theta_star).sum(dim=1)

            r_t = torch.bernoulli(torch.sigmoid(z_t))

            # True regret
            rew = torch.sigmoid(z_t)
            regrets[:, t] = s_beta - rew

            # Append history
            Phi_hist = torch.cat([Phi_hist, phi_t.unsqueeze(1)], dim=1)
            r_hist   = torch.cat([r_hist, r_t.unsqueeze(1)], dim=1)

        all_regrets[exp_idx:exp_idx+B] = regrets
        exp_idx += B

    # Save
    avg = all_regrets.mean(dim=0)
    torch.save(all_regrets.cpu(), file_all)
    torch.save(avg.cpu(), file_avg)

    print(f"Saved contextual results to {file_all}")
    return avg, all_regrets



# ------------------------ runs experiments on multiple betas -------------------------
def sweep_betas(
    betas=None,
    d: int = 10,
    T: int = 200,
    num_exp: int = 64, #1024,
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
        avg, runs = run_contextual_logistic_bandits_exp(
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