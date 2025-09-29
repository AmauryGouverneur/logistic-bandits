# plots_ts.py
import os
import math
from typing import Sequence, Tuple

import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import matplot2tikz as tikzplotlib

DARKORANGE = (255/255, 127/255, 14/255)      # darkorange25512714
STEELBLUE = (31/255, 119/255, 180/255)      # steelblue31119180
FORESTGRN = (44/255, 160/255,  44/255)      # forestgreen4416044
PURPLE    = (116/255, 72/255, 155/255)      # purple11672155
LIGHTGRAY = (204/255, 204/255, 204/255)     # lightgray204204204
DARK      = (0.15, 0.15, 0.15)


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


# ---------- bounds ----------

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def phi_beta(x, beta):
    return sigmoid(beta * x)

def f_ratio(x, beta):
    """ f(x) = [phi_beta(1) - phi_beta(1 - x)] / x with the x->0+ limit. """
    if x <= 0:
        s = sigmoid(beta)
        return beta * s * (1.0 - s)  # limit as x->0+
    num = phi_beta(1.0, beta) - phi_beta(1.0 - x, beta)
    return num / x

def _delta_beta(beta):
    # maximize f over [0, 2] by minimizing -f with a bounded scalar solver
    obj = lambda x: -f_ratio(x, beta)
    res = minimize_scalar(obj, bounds=(0.0, 2.0), method="bounded", options={"xatol": 1e-9})
    # Check endpoints just in case
    candidates = np.array([0.0, res.x, 2.0])
    vals = np.array([f_ratio(c, beta) for c in candidates])
    i = np.argmax(vals)
    return float(candidates[i])


def _bound_russo_van_roy(beta: float, d: int, T, gamma: float = 1.0,
                                 C: float = 1.0, sigma: float = 0.5, dimK: int | None = None):
    """
    Explicit TS regret upper bound for logistic bandits (Russo & Van Roy, Propositon 10 + Propositon 12),
    specialized with C=1 (Bernoulli rewards), sigma=1/2, and dimK(F) ≤ d by default.

    R(T) <= 1+ (dimE(F,T^{-1}) + 1) C + 16 σ sqrt(dimE(F,T^{-1})*(1 + o(1) + dimK)log(T)T

    Parameters
    ----------
    beta : slope parameter in sigmoid(beta * <theta,phi>)
    d    : dimension
    T    : scalar or 1D array of horizons
    gamma : bound s.t. ||phi(a)|| ≤ gamma  
    C    : reward bound (Bernoulli ⇒ 1)
    sigma: sub-Gaussian parameter (Bernoulli ⇒ 1/2)
    dimK : Kolmogorov dimension upper bound (defaults to d)

    Returns
    -------
    np.ndarray of the same shape as T with the bound values.
    """
    T = np.asarray(T, dtype=float)
    dimK = d if dimK is None else int(dimK)

    # derivative bounds over x ∈ [-S*gamma, S*gamma]
    E = np.exp(beta * gamma)
    h_min = beta * E / (1.0 + E) ** 2     # underline h
    r = (1.0 + E) ** 2 / (4.0 * E)

    # explicit eluder-dimension bound at epsilon = 1/T (Prop. 12):
    # dimE ≤ (3 d r^2 e/(e-1)) * ln( (3r/2) * [1 + (2 S h_min T)^2] ) + 1
    const = (3.0 * d * (r ** 2) * np.e) / (np.e - 1.0)
    inside = (3.0 * r**2) * (1.0 + (2.0 * h_min * T) ** 2)
    dim_E = const * np.log(inside) + 1.0

    # Prop. 10 with C=1, sigma=1/2, dimK ≤ d:
    # Reg ≤ 1 + (dim_E + 1) * C + 16*sigma * sqrt( D_E * (1 + dimK) * log T * T )
    # use max(T, e) to keep log positive for small T
    logT = np.log(np.maximum(T, np.e))
    bound = 1.0 + (dim_E + 1.0) * C + 16.0 * sigma * np.sqrt(dim_E * (1.0 + dimK) * logT * T)
    return bound




def _bound_dong_van_roy(beta: float, d: int, T_vec: np.ndarray) -> np.ndarray:
    """
    Dong & Van Roy (2018) as a function of T (vectorized).

    Let ε(T) = d (1+e^β)^2 / (4 e^β √(2T)).

    If ε(T) < 1:
        R(T) = [d (1+e^β)^2 / (4 e^β)] * √{ T * ln( 3 + [24 √(2T) e^β] / [d (1+e^β)^2] ) }
    else:
        R(T) = [d (1+e^β)^2 / (4 e^β)] * √T  +  T
    """
    T_vec = np.asarray(T_vec, dtype=float)
    eb = np.exp(beta)
    c1 = d * (1.0 + eb)**2 / (4.0 * eb)

    # epsilon(T)
    eps = c1 / np.sqrt(2.0 * T_vec)  # = d(1+e^β)^2 / (4 e^β √(2T))

    # branch 1: ε < 1
    inside_log = 3.0 + (24.0 * np.sqrt(2.0 * T_vec) * eb) / (d * (1.0 + eb)**2)
    r_log = c1 * np.sqrt(T_vec * np.log(inside_log))

    # branch 2: ε >= 1
    r_linear = c1 * np.sqrt(T_vec) + T_vec

    return np.where(eps < 1.0, r_log, r_linear)


def _bound_ours(beta: float, d: int, T_vec: np.ndarray) -> np.ndarray:
    """
    Our bound as a function of T:
    R(T) <= [2 / δ(β)] * √{ d * T * ( d ln(1 + 2/ε) + (ε^2 β^2 T)/2 ) }

    Let ε = (1/β) * √( (2d) / T ).

    If ε < 1:
        R(T) <= [2d / δ(β)] * √{ T * ln( 3 + 6 β √T / √(2d) ) }
    else:
        R(T) <= [2d / δ(β)] * √{ T * ( 1 + (β^2 / 2) T ) }

    Note: for small values of (T β^2) / (2d), the bound with
    ε_small = ((2d) / (3 T β^2))^(2/5) (if ε_small <=1) gives improved results:
        R(T) <= [2 / δ(β)] * √{ d * T * ( d ln(1 + 2/ε_small) + (ε_small^2 β^2 T)/2 ) }
    and take the minimum of the two bounds.
    """
    T_vec = np.asarray(T_vec, dtype=float)
    delta = _delta_beta(beta)
    pref = 2.0 * d / delta

    # epsilon(T) for large values of (T beta^2) / (2d)
    eps = (1.0 / beta) * np.sqrt((2.0 * d) / T_vec)

    inside_log = 3.0 + 6.0 * beta * np.sqrt(T_vec) / np.sqrt(2.0 * d)
    r_log = pref * np.sqrt(T_vec * np.log(inside_log))

    # epsilon(T) for small values of (T beta^2) / (2d)
    eps_small = ((2.0* d)/( 3.0*T_vec *beta**2 ))**(2/5)  # not used directly
    r_log_small = 2.0 / delta* np.sqrt(d*T_vec * (d * np.log(1+2.0/eps_small)+1/2*eps_small**2*beta**2*T_vec)) 

    # branch: 
    r_quad = pref * np.sqrt(T_vec * (1.0 + 0.5 * (beta**2) * T_vec))

    r_log = np.where(eps < 1.0, r_log, r_quad)
    r_log_small = np.where(eps_small < 1.0, r_log_small, r_quad)

    return np.minimum(r_log, r_log_small)


# ---------- plots ----------



def plot_cumulative_regret_two_betas_with_ci(
    d: int = 10,
    T: int = 200,
    beta_solid: float = 2.0,   # solid line
    beta_dashed: float = 4.0,  # dashed line
    save_dir: str = "results_experiments",
    figdir: str = "figures",
    fig_basename: str = "regret_b2_b4_with_ci",
    alpha_ci: float = 0.05,
    log_y: bool = True,
):
    """
    Plot cumulative regret for two betas on the same plot:
      - beta_solid: solid line
      - beta_dashed: dashed line
    Each with mean ± CI shaded region.
    Saves figures/<fig_basename>.png and .tex
    """
    os.makedirs(figdir, exist_ok=True)

    # Load runs (expects your load_runs + _cum_stats utils)
    runs_solid = load_runs(beta_solid, d, save_dir)
    runs_dashed = load_runs(beta_dashed, d, save_dir)
    if runs_solid is None or runs_dashed is None:
        missing = []
        if runs_solid is None:  missing.append(beta_solid)
        if runs_dashed is None: missing.append(beta_dashed)
        raise FileNotFoundError(f"No saved results for beta(s): {missing}")

    # Match horizons (and clamp to requested T)
    T_use = min(T, runs_solid.shape[1], runs_dashed.shape[1])
    runs_solid  = runs_solid[:, :T_use]
    runs_dashed = runs_dashed[:, :T_use]

    # Cumulative means & CIs
    mean_solid,  lo_solid,  hi_solid  = _cum_stats(runs_solid)
    mean_dashed, lo_dashed, hi_dashed = _cum_stats(runs_dashed)

    x = np.arange(1, T_use + 1, dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4.2))

    # β_solid — solid line
    ax.plot(x, mean_solid.cpu().numpy(), linestyle='-', color=PURPLE, linewidth=2.0)
    ax.fill_between(x, lo_solid.cpu().numpy(), hi_solid.cpu().numpy(), alpha=0.20, color=PURPLE)

    # β_dashed — dashed line
    ax.plot(x, mean_dashed.cpu().numpy(), linestyle='--', color=PURPLE, linewidth=2.0)
    ax.fill_between(x, lo_dashed.cpu().numpy(), hi_dashed.cpu().numpy(), alpha=0.20, color=PURPLE)

    if log_y:
        ax.set_yscale('log')
    ax.set_xlabel("T")
    ax.set_ylabel("Regret" + (" (log scale)" if log_y else ""))
    ax.grid(True, which='both', linewidth=0.3, alpha=0.4)

    # -------- Legend #1: which beta (line styles) --------
    handles_beta = [
        Line2D([0], [0], color=PURPLE, lw=2.0, linestyle='-',  label=fr"TS ($\beta={beta_solid}$)"),
        Line2D([0], [0], color=PURPLE, lw=2.0, linestyle='--', label=fr"TS ($\beta={beta_dashed}$)"),
    ]
    leg1 = ax.legend(handles=handles_beta, loc='upper left', frameon=True)
    ax.add_artist(leg1)  # keep when adding second legend

    # -------- Legend #2: what styles mean (mean vs CI) --------
    mean_proxy = Line2D([0], [0], color=PURPLE, lw=2.0, linestyle='-')
    ci_proxy   = Patch(facecolor=PURPLE, alpha=0.20, edgecolor='none')
    leg2 = ax.legend(
        handles=[mean_proxy, ci_proxy],
        labels=["Mean", f"{int((1 - alpha_ci)*100)}% CI (shaded)"],
        loc='lower right',
        frameon=True,
    )

    plt.tight_layout()
    _save_png_and_tikz(fig_basename, figdir=figdir)



def plot_final_cumulative_regret_vs_beta_with_ci(
    betas: Sequence[float] = (0.25, 0.5, 1.0, 1.5, 2.0, *range(3, 11)),
    d: int = 10,
    T: int = 200,
    save_dir: str = "results_experiments",
    figdir: str = "figures",
    fig_basename: str = "regret_T200_vs_beta_with_ci",
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
    plt.plot(beta_np, mean_np, linestyle='-', label=f"Thompson Sampling (mean)", color=PURPLE, linewidth=2.0)
    plt.fill_between(beta_np, lo_np, hi_np, alpha=0.20, label=f"{int((1-alpha_ci)*100)}% CI", color=PURPLE)
    if log_y:
        plt.yscale('log')
    plt.xlabel(r"$\beta$")
    plt.ylabel(f"Regret at T={T}" + (" (log scale)" if log_y else ""))
    plt.grid(True, which='both', linewidth=0.3, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    _save_png_and_tikz(fig_basename, figdir=figdir)




def plot_cumulative_regret_with_bounds_two_betas(
    d: int = 10,
    T: int = 200,
    beta1: float = 2.0,   # solid
    beta2: float = 4.0,   # dashed
    save_dir: str = "results_experiments",
    figdir: str = "figures",
    fig_basename: str = "regret_with_bounds_b2_b4",
    log_y: bool = True,
):
    """
    Superposed plot:
      - β=2.0: Thompson Sampling + bounds (solid lines)
      - β=4.0: Thompson Sampling + bounds (dashed lines)
    Colors:
      - Dong & Van Roy (2018) bound: steel blue
      - Our bound: forest green
      - Empirical TS mean: purple
    Grid: light gray, behind the curves.
    Saves figures/<fig_basename>.png and .tex (via _save_png_and_tikz)
    """
    os.makedirs(figdir, exist_ok=True)

    # Load runs
    runs1 = load_runs(beta1, d, save_dir)
    runs2 = load_runs(beta2, d, save_dir)
    if runs1 is None or runs2 is None:
        missing = []
        if runs1 is None: missing.append(beta1)
        if runs2 is None: missing.append(beta2)
        raise FileNotFoundError(f"No saved results for beta(s): {missing}")

    # Match horizons (and clamp to requested T)
    T_use = min(T, runs1.shape[1], runs2.shape[1])
    runs1 = runs1[:, :T_use]
    runs2 = runs2[:, :T_use]

    # Empirical cumulative means
    mean_cum1 = runs1.cumsum(dim=1).mean(dim=0).cpu().numpy()
    mean_cum2 = runs2.cumsum(dim=1).mean(dim=0).cpu().numpy()
    x = np.arange(1, T_use + 1, dtype=float)

    # Bounds (vectorized in T)
    b1_rvr  = _bound_russo_van_roy(beta1, d, x)
    b1_dvr  = _bound_dong_van_roy(beta1, d, x)
    b1_ours = _bound_ours(beta1, d, x)
    b2_rvr  = _bound_russo_van_roy(beta2, d, x)
    b2_dvr  = _bound_dong_van_roy(beta2, d, x)
    b2_ours = _bound_ours(beta2, d, x)

    # Figure + grid behind content
    plt.figure(figsize=(7.6, 4.8))
    ax = plt.gca()
    ax.set_axisbelow(True)  # ensures grid is behind lines
    plt.grid(True, color=LIGHTGRAY, linewidth=0.6)

    lw_emp = 2.0
    lw_bnd = 1.8

    # β = 2.0 — solid
    plt.plot(x, mean_cum1, linestyle='-', color=PURPLE,    linewidth=lw_emp)
    plt.plot(x, b1_rvr,  linestyle='-', color=DARKORANGE, linewidth=lw_bnd)
    plt.plot(x, b1_dvr,   linestyle='-', color=STEELBLUE,  linewidth=lw_bnd)
    plt.plot(x, b1_ours,  linestyle='-', color=FORESTGRN,  linewidth=lw_bnd)

    # β = 4.0 — dashed
    plt.plot(x, mean_cum2, linestyle='--', color=PURPLE,    linewidth=lw_emp)
    plt.plot(x, b2_rvr,  linestyle='--', color=DARKORANGE, linewidth=lw_bnd)
    plt.plot(x, b2_dvr,   linestyle='--', color=STEELBLUE,  linewidth=lw_bnd)
    plt.plot(x, b2_ours,  linestyle='--', color=FORESTGRN,  linewidth=lw_bnd)

    # Markers at t = T_use (last point), hollow (no fill)
    xT = x[-1]
    ms = 6  # marker size

    # β = 2.0 — circle markers (hollow)
    plt.plot([xT], [mean_cum1[-1]], marker='o', mfc='none', mec=PURPLE,    mew=1.5, ms=ms, linestyle='None')
    plt.plot([xT], [b1_rvr[-1]],  marker='o', mfc='none', mec=DARKORANGE, mew=1.5, ms=ms, linestyle='None')
    plt.plot([xT], [b1_dvr[-1]],    marker='o', mfc='none', mec=STEELBLUE, mew=1.5, ms=ms, linestyle='None')
    plt.plot([xT], [b1_ours[-1]],   marker='o', mfc='none', mec=FORESTGRN, mew=1.5, ms=ms, linestyle='None')

    # β = 4.0 — diamond markers (hollow)
    plt.plot([xT], [mean_cum2[-1]], marker='D', mfc='none', mec=PURPLE,    mew=1.5, ms=ms, linestyle='None')
    plt.plot([xT], [b2_rvr[-1]],  marker='D', mfc='none', mec=DARKORANGE, mew=1.5, ms=ms, linestyle='None')
    plt.plot([xT], [b2_dvr[-1]],    marker='D', mfc='none', mec=STEELBLUE, mew=1.5, ms=ms, linestyle='None')
    plt.plot([xT], [b2_ours[-1]],   marker='D', mfc='none', mec=FORESTGRN, mew=1.5, ms=ms, linestyle='None')

    # --- after you've drawn all six curves and the hollow markers ---

    ax = plt.gca()

    # Legend A: color → curve type
    color_handles = [
        Line2D([0], [0], color=PURPLE,    lw=2.0, linestyle='-', label="Thompson Sampling"),
        Line2D([0], [0], color=DARKORANGE, lw=2.0, linestyle='-', label="Russo & Van Roy (2014)"),
        Line2D([0], [0], color=STEELBLUE, lw=2.0, linestyle='-', label="Dong & Van Roy (2018)"),
        Line2D([0], [0], color=FORESTGRN, lw=2.0, linestyle='-', label="This paper"),
    ]
    leg_colors = ax.legend(handles=color_handles, loc="lower right",
                        frameon=False)

    # Legend B: linestyle → beta
    DARK = (0.15, 0.15, 0.15)
    style_handles = [
        Line2D([0], [0], color=DARK, lw=2.0, linestyle='-', label=r"$\beta=2.0$"),
        Line2D([0], [0], color=DARK, lw=2.0, linestyle='--', label=r"$\beta=4.0$"),
    ]
    leg_styles = ax.legend(handles=style_handles, loc="upper left",
                        frameon=False)

    # Make both show (second legend would otherwise replace the first)
    ax.add_artist(leg_colors)


    if log_y:
        plt.yscale('log')

    plt.xlabel("T")
    plt.ylabel("Regret" + (" (log-scale)" if log_y else ""))
    #plt.legend(ncol=2)
    plt.tight_layout()

    _save_png_and_tikz(fig_basename, figdir=figdir)


def plot_final_regret_vs_beta_with_bounds(
    betas = np.r_[0.25:4.0+0.25:0.25,  4.5:10.0+0.5:0.5].tolist(),
    d: int = 10,
    T: int = 200,
    save_dir: str = "results_experiments",
    figdir: str = "figures",
    fig_basename: str = "regret_T200_vs_beta_bounds",
    log_y: bool = True,
    skip_missing: bool = True,
):
    """
    Cumulative regret at time T vs β, with bounds:
    Colors:
      - Russo & Van Roy (2014) [dark orange]
      - Dong & Van Roy (2019) [steel blue]
      - Ours [forest green]
      - TS Empirical mean [purple]
    Adds hollow markers at β=2.0 (circle) and β=4.0 (diamond) for all curves.
    """
    os.makedirs(figdir, exist_ok=True)

    beta_vals, emp_vals, rvr_vals, dvr_vals, ours_vals = [], [], [], [], []
    for b in betas:
        b = float(b)
        runs = load_runs(b, d, save_dir)
        if runs is None:
            msg = f"[skip] no saved runs for beta={b}"
            if skip_missing:
                print(msg); continue
            raise FileNotFoundError(msg)

        T_use = min(T, runs.shape[1])
        cum_T = runs[:, :T_use].cumsum(dim=1)[:, -1]  # (N,)
        emp_mean = float(cum_T.mean().item())

        # bounds at scalar T
        rvr = float(_bound_russo_van_roy(b, d, T_use))
        dvr  = float(_bound_dong_van_roy(b, d, T_use))
        ours = float(_bound_ours(b, d, T_use))

        beta_vals.append(b)
        emp_vals.append(emp_mean)
        rvr_vals.append(rvr)
        dvr_vals.append(dvr)
        ours_vals.append(ours)

    if not beta_vals:
        print("Nothing to plot (no betas loaded).")
        return

    beta_np = np.array(beta_vals, dtype=float)
    emp_np  = np.array(emp_vals,  dtype=float)
    rvr_np  = np.array(rvr_vals,  dtype=float)
    dvr_np  = np.array(dvr_vals,  dtype=float)
    ours_np = np.array(ours_vals, dtype=float)

    plt.figure(figsize=(7.6, 4.8))
    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.grid(True, color=LIGHTGRAY, linewidth=0.6)

    # Lines
    emp_line,  = plt.plot(beta_np, emp_np,  color=PURPLE,    linewidth=2.0)
    rvr_line,  = plt.plot(beta_np, rvr_np,  color=DARKORANGE, linewidth=1.8)
    dvr_line,  = plt.plot(beta_np, dvr_np,  color=STEELBLUE, linewidth=1.8)
    ours_line, = plt.plot(beta_np, ours_np, color=FORESTGRN, linewidth=1.8)

    # Hollow markers at β=2.0 (circle) and β=4.0 (diamond)
    ms, mew = 7, 1.5
    def _mark_at_beta(beta_target, marker, color, y_vals):
        # find exact match if present; otherwise do nothing
        idx = np.where(np.isclose(beta_np, beta_target))[0]
        if idx.size > 0:
            xT, yT = beta_np[idx[0]], y_vals[idx[0]]
            plt.plot([xT], [yT], marker=marker, mfc='none', mec=color, mew=mew, ms=ms, linestyle='None')

    for arr, col in [(emp_np, PURPLE), (rvr_np, DARKORANGE), (dvr_np, STEELBLUE), (ours_np, FORESTGRN)]:
        _mark_at_beta(2.0, 'o', col, arr)  # circle at β=2.0
        _mark_at_beta(4.0, 'D', col, arr)  # diamond at β=4.0

    if log_y:
        plt.yscale('log')

    plt.xlabel(r"$\beta$")
    plt.ylabel(f"Regret at T={T}" + (" (log scale)" if log_y else ""))

    # Legend: color → curve type (proxy handles)
    color_handles = [
        Line2D([0], [0], color=PURPLE,    lw=2.0, linestyle='-', label="Thompson Sampling"),
        Line2D([0], [0], color=DARKORANGE, lw=1.8, linestyle='-', label="Russo & Van Roy (2014)"),
        Line2D([0], [0], color=STEELBLUE, lw=1.8, linestyle='-', label="Dong & Van Roy (2018)"),
        Line2D([0], [0], color=FORESTGRN, lw=1.8, linestyle='-', label="This paper"),
    ]
    leg_colors = ax.legend(handles=color_handles, loc="upper left",
                           frameon=False)

    ax.add_artist(leg_colors)

    plt.tight_layout()
    _save_png_and_tikz(fig_basename, figdir=figdir)

# ---------- convenience wrappers ----------


def run_plot_final_regret_vs_beta_with_ci():
    plot_final_cumulative_regret_vs_beta_with_ci(
        betas=np.r_[0.25:4.0+0.25:0.25,  4.5:10.0+0.5:0.5].tolist(),
        d=10, T=200,
        save_dir="results_experiments",
        figdir="figures",
        fig_basename="regret_T200_vs_beta_with_ci",
        alpha_ci=0.05,
        log_y=True,
    )

def run_plot_two_betas_with_ci():
    plot_cumulative_regret_two_betas_with_ci(
        d=10, T=200,
        beta_solid=2.0, beta_dashed=4.0,
        save_dir="results_experiments",
        figdir="figures",
        fig_basename="regret_b2_b4_with_ci",
        alpha_ci=0.05,
        log_y=True,
    )

def run_plot_with_bounds_two_betas():
    plot_cumulative_regret_with_bounds_two_betas(
        d=10, T=200,
        beta1=2.0, beta2=4.0,
        save_dir="results_experiments",
        figdir="figures",
        fig_basename="regret_with_bounds_b2_b4",
        log_y=True,
    )

def run_plot_final_regret_vs_beta():
    plot_final_regret_vs_beta_with_bounds(
        betas=np.r_[0.25:4.0+0.25:0.25,  4.5:10.0+0.5:0.5].tolist(),
        d=10, T=200,
        save_dir="results_experiments",
        figdir="figures",
        fig_basename="regret_T200_vs_beta_bounds",
        log_y=True,
        skip_missing=True,
    )

if __name__ == "__main__":
    run_plot_final_regret_vs_beta_with_ci()
    run_plot_two_betas_with_ci()
    run_plot_with_bounds_two_betas()
    run_plot_final_regret_vs_beta()
