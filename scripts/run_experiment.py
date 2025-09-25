import argparse, os, torch
from logistic_bandit_ts import run_logistic_bandits_TS_exp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=10)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--num-exp", type=int, default=100)
    ap.add_argument("--kappa", type=float, default=8.0)
    ap.add_argument("--mh-steps", type=int, default=100)
    ap.add_argument("--chains", type=int, default=None, help="parallel MH chains (auto: 128 on CUDA, 1 on CPU)")
    ap.add_argument("--out", type=str, default="results_experiments/logistic_ts_beta_{beta}_d_{d}.pt")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    path = args.out.format(beta=args.beta, d=args.d)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    expected = run_logistic_bandits_TS_exp(
        d=args.d, beta=args.beta, T=args.T,
        num_exp=args.num_exp, kappa=args.kappa, mh_steps=args.mh_steps,
        progress=True, chains=args.chains
    )
    torch.save(expected, path)
    print("Saved:", path)

if __name__ == "__main__":
    main()
