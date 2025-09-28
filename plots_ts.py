# plots_ts.py
import os
import math
from typing import Sequence, Tuple

import torch
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplot2tikz as tikzplotlib



# ---------- basic stats ----------

def mean_and_ci(all_runs: torch.Tensor, alpha: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    all_runs: (num_exp, T) tensor
    returns mean, lower, upper each shape (T,)
    """
    mean = all_runs.mean(dim=0)
    n = max(1, all_runs.shape[0])
    se = all_runs.std(dim=0, unbiased=True) / math.sqrt(n) if n > 1 else torch.zeros_like(mean)
    z = float(norm.ppf(1 - alpha / 2))
    lower = mean - z * se
    upper = mean + z * se
    return mean, lower, upper


# ---------- load saved runs ----------

def load_runs(beta: float, d: int, save_dir: str = "results_experiments") -> torch.Tensor:
    """
    Load per-experiment regrets. Prefers *_batched.pt, falls back to plain.
    Returns CPU tensor of shape (num_exp, T). Raises FileNotFoundError if missing.
    """
    path_batched = os.path.join(save_dir, f"logistic_ts_all_beta_{beta}_d_{d}_batched.pt")
    path_plain   = os.path.join(save_dir, f"logistic_ts_all_beta_{beta}_d_{d}.pt")
    path = path_batched if os.path.exists(path_batched) else path_plain
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved runs for beta={beta} (looked for {path_batched} and {path_plain})")
    return torch.load(path, map_location="cpu")


# ---------- helpers ----------

def _cum_stats(runs_2d: torch.Tensor):
    """Per-step regret -> cumulative regret, then mean & CI."""
    cum = runs_2d.cumsum(dim=1)
    return mean_and_ci(cum)


def _ci_1d(samples: torch.Tensor, alpha: float = 0.05):
    """CI for 1D samples (N,). Returns mean, lo, hi (0-dim tensors)."""
    n = samples.numel()
    mean = samples.mean()
    se = samples.std(unbiased=True) / math.sqrt(n) if n > 1 else torch.zeros((), dtype=mean.dtype, device=mean.device)
    z = float(norm.ppf(1 - alpha / 2))
    return mean, mean - z * se, mean + z * se


def _ensure_figdir(figdir: str = "figures"):
    os.makedirs(figdir, exist_ok=True)


def _save_png_and_tikz(fig_basename: str, figdir: str = "figures", dpi: int = 150):
    """
    Save current matplotlib figure as PNG and TikZ under figures/<basename>.(png|tex)
    fig_basename should NOT include extension.
    """
    _ensure_figdir(figdir)
    png_path = os.path.join(figdir, f"{fig_basename}.png")
    tex_path = os.path.join(figdir, f"{fig_basename}.tex")
    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    tikzplotlib.save(tex_path)
    print(f"Saved: {png_path}\nSaved: {tex_path}")


# ---------- plots ----------

def plot_cumulative_regret_two_betas(
    d: int = 10,
    T: int = 200,
    beta_solid: float = 2.0,
    beta_dashed: float = 4.0,
    save_dir: str = "results_experiments",
    figdir: str = "figures",
    fig_basename: str = "cumreg_b2_b4",
    alpha_ci: float = 0.05,
    log_y: bool = True,
):
    """
    Cumulative regret vs t.
    - β=2.0 solid, β=4.0 dashed
    - 95% CI bands (configurable via alpha_ci)
    - y-axis log if log_y=True
    Saves figures/<fig_basename>.png and .tex
    """
    runs2 = load_runs(beta_solid, d, save_dir)
    runs4 = load_runs(beta_dashed, d, save_dir)

    # match time horizons
    T_use = min(T, runs2.shape[1], runs4.shape[1])
    runs2 = runs2[:, :T_use]
    runs4 = runs4[:, :T_use]

    mean2, lo2, hi2 = _cum_stats(runs2)
    mean4, lo4, hi4 = _cum_stats(runs4)

    x = torch.arange(1, T_use + 1).cpu().numpy()
    m2, l2, h2 = (t.cpu().numpy() for t in (mean2, lo2, hi2))
    m4, l4, h4 = (t.cpu().numpy() for t in (mean4, lo4, hi4))

    plt.figure(figsize=(7, 4.5))
    plt.plot(x, m2, label=fr"$\beta={beta_solid}$", linestyle='-')
    plt.fill_between(x, l2, h2, alpha=0.20)
    plt.plot(x, m4, label=fr"$\beta={beta_dashed}$", linestyle='--')
    plt.fill_between(x, l4, h4, alpha=0.20)
    if log_y:
        plt.yscale('log')
    plt.xlabel("t")
    plt.ylabel("Cumulative regret" + (" (log scale)" if log_y else ""))
    plt.legend()
    plt.tight_layout()
    _save_png_and_tikz(fig_basename, figdir=figdir)


def plot_final_cumulative_regret_vs_beta(
    betas: Sequence[float] = (0.25, 0.5, 1.0, 1.5, 2.0, *range(3, 11)),
    d: int = 10,
    T: int = 200,
    save_dir: str = "results_experiments",
    figdir: str = "figures",
    fig_basename: str = "cumreg_T_vs_beta",
    alpha_ci: float = 0.05,
    log_y: bool = True,
):
    """
    For each beta, load (num_exp, T) regrets, take cumulative at time T,
    then plot mean ± CI vs beta. y-axis log if log_y=True.
    Saves figures/<fig_basename>.png and .tex
    """
    beta_vals, means, los, his = [], [], [], []
    for b in betas:
        runs = load_runs(float(b), d, save_dir)
        T_use = min(T, runs.shape[1])
        cum_T = runs[:, :T_use].cumsum(dim=1)[:, -1].to(dtype=torch.float64)
        m, lo, hi = _ci_1d(cum_T.cpu(), alpha=alpha_ci)
        beta_vals.append(float(b))
        means.append(float(m))
        los.append(float(lo))
        his.append(float(hi))

    import numpy as np
    beta_np = np.array(beta_vals, dtype=float)
    mean_np = np.array(means, dtype=float)
    lo_np   = np.array(los, dtype=float)
    hi_np   = np.array(his, dtype=float)

    plt.figure(figsize=(7, 4.2))
    plt.plot(beta_np, mean_np, linestyle='-', label=f"Cumulative regret at T={T}")
    plt.fill_between(beta_np, lo_np, hi_np, alpha=0.20, label=f"{int((1-alpha_ci)*100)}% CI")
    if log_y:
        plt.yscale('log')
    plt.xlabel(r"$\beta$")
    plt.ylabel(f"Cumulative regret at T={T}" + (" (log scale)" if log_y else ""))
    plt.grid(True, which='both', linewidth=0.3, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    _save_png_and_tikz(fig_basename, figdir=figdir)


# ---------- convenience wrappers ----------

def run_plot_final_cumreg_vs_beta():
    plot_final_cumulative_regret_vs_beta(
        betas=[0.25, 0.5, 1.0, 1.5, 2.0] + list(range(3, 11)),
        d=10, T=200,
        save_dir="results_experiments",
        figdir="figures",
        fig_basename="cumreg_T200_vs_beta",
        alpha_ci=0.05,
        log_y=True,
    )

def run_plot_two_betas():
    plot_cumulative_regret_two_betas(
        d=10, T=200,
        beta_solid=2.0, beta_dashed=4.0,
        save_dir="results_experiments",
        figdir="figures",
        fig_basename="cumreg_b2_b4",
        alpha_ci=0.05,
        log_y=True,
    )

if __name__ == "__main__":
    run_plot_final_cumreg_vs_beta()
