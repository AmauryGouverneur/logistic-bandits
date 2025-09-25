# Logistic Bandits with Thompson Sampling (vMF–MH on the Sphere)

This repo runs a logistic bandit with actions and parameters on the unit sphere.
Posterior sampling uses Metropolis–Hastings on \(\mathbb S^{d-1}\).

## Quickstart

```bash
conda env create -f env/environment.cuda.yml
conda activate logistic-bandits-cuda
python scripts/run_experiment.py --config config/default.yaml
python scripts/plot_results.py --path results_experiments/logistic_ts_beta_2.0_d_10.pt
