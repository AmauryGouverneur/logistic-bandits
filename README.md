# Logistic Bandits with Thompson Sampling (vMF–MH on the Sphere)

This repo runs a logistic bandit with actions and parameters on the unit sphere.
Posterior sampling uses an exactly symmetric von Mises–Fisher (vMF) random-walk
Metropolis–Hastings on \(\mathbb S^{d-1}\).

## Quickstart

### Option A: Conda (CPU)
```bash
mamba env create -f env/environment.cpu.yml  # or: conda env create ...
conda activate logistic-bandits
python scripts/run_experiment.py --config config/default.yaml
python scripts/plot_results.py --path results_experiments/logistic_ts_beta_2.0_d_10.pt
