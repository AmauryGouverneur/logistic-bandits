# sweep_betas.py
from logistic_bandit_ts import run_logistic_bandits_TS_exp_batched
BETAS = [0.25, 0.5, 1.0, 1.5, 2.0] + list(range(3, 11))  # 3,4,...,10
D, T = 10, 200
NUM_EXP = 120          # total experiments per beta
BATCH_SIZE = 48        # run this many experiments in parallel on GPU
CHAINS = 128           # per-experiment MH chains
MH_STEPS = 8           # MH steps per round (persistent chains)

for beta in BETAS:
    avg, runs = run_logistic_bandands_TS_exp_batched(
        d=D, beta=float(beta), T=T, num_exp=NUM_EXP,
        batch_size=BATCH_SIZE, chains=CHAINS, mh_steps=MH_STEPS,
        progress=True, append=True,
    )
    print(f"β={beta}: saved {runs.shape}")