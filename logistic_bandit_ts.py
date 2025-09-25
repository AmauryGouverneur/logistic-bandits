import os
import torch
import matplotlib.pyplot as plt
from mh_sphere import MHSphereSampler
from tqdm import trange

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def phi(x, beta):
    return torch.sigmoid(beta * x)

def sample_reward(A, Theta, beta):
    # Bernoulli reward with P=phi(<A,Theta>, beta)
    p = phi(torch.dot(A, Theta), beta).clamp(0.0, 1.0)
    return torch.bernoulli(p)

def sample_prior_theta(d):
    X = torch.randn(d, device=device)
    return X / X.norm()

def log_posterior(theta, history_actions, history_rewards, beta):
    """
    Uniform prior on the sphere + logistic likelihood.
    theta: (d,), unit
    history_actions: list[(d,)]
    history_rewards: list[scalar]
    """
    if not history_actions:
        return torch.tensor(0.0, device=theta.device)
    A = torch.stack(history_actions, dim=0)             # (t, d)
    r = torch.stack(history_rewards, dim=0).to(theta)   # (t,)
    z = (A @ theta) * beta                              # (t,)
    # sum r*z - log(1+exp(z)) with stable softplus
    return (r * z - torch.nn.functional.softplus(z)).sum()

def thompson_sampled_action(history_actions, history_rewards, beta,
                            sampler: MHSphereSampler,
                            mh_steps=100, chains=None, return_batch=False):
    """
    If CUDA is available: run multiple parallel MH chains and return one draw.
    If return_batch=True, also return the whole batch of posterior samples (K,d).
    """
    d = history_actions[0].numel() if history_actions else sampler.d
    # warm start
    theta0 = (history_actions[-1].detach().to(device)
              if history_actions else sample_prior_theta(d).to(device))

    # cache data for fast batched logp
    if history_actions:
        A = torch.stack(history_actions, dim=0).to(device)   # (t,d)
        r = torch.stack(history_rewards, dim=0).to(device)   # (t,)
    else:
        A = torch.empty(0, d, device=device); r = torch.empty(0, device=device)

    def logp(th):
        z = (A @ th) * beta
        return (r * z - torch.nn.functional.softplus(z)).sum()

    def logp_batch(Theta):
        if A.numel() == 0:
            return torch.zeros(Theta.shape[0], device=Theta.device, dtype=Theta.dtype)
        Z = beta * (A @ Theta.T)                 # (t,K)
        return (r[:, None] * Z - torch.nn.functional.softplus(Z)).sum(dim=0)  # (K,)

    use_cuda = (device.type == 'cuda')
    K = chains if chains is not None else (128 if use_cuda else 1)

    if K == 1:
        theta_sample, _ = sampler.step(theta0, logp, n_steps=mh_steps)
        return (theta_sample, theta_sample.unsqueeze(0)) if return_batch else theta_sample

    # Multi-chain
    mu0 = theta0 / theta0.norm()
    Theta0 = sampler._vmf_sample_batch(mu0.expand(K, -1))          # (K,d)
    ThetaT, _ = sampler.step_batch(Theta0, logp_fn_batch=logp_batch, n_steps=mh_steps)
    idx = torch.randint(0, K, (1,), device=ThetaT.device).item()
    return (ThetaT[idx], ThetaT) if return_batch else ThetaT[idx]


def bandit_experiment(d, beta, T, sampler: MHSphereSampler, mh_steps=100, chains=None):
    history_actions, history_rewards = [], []
    expected_regret = torch.zeros(T, 1, device=device)

    theta_star = sample_prior_theta(d)
    action = sample_prior_theta(d)
    reward = sample_reward(action, theta_star, beta)
    history_actions.append(action); history_rewards.append(reward)

    opt_reward = phi(torch.tensor(1.0, device=device), beta)  # phi(beta)

    for t in range(T):
        # get both: the action we will play, and the full posterior sample batch
        action, Theta_batch = thompson_sampled_action(
            history_actions, history_rewards, beta,
            sampler=sampler, mh_steps=mh_steps, chains=chains, return_batch=True
        )

        # interact with env using one TS action (unchanged algorithm)
        reward = sample_reward(action, theta_star, beta)
        history_actions.append(action); history_rewards.append(reward)

        # Rao–Blackwellized (low-variance) expected regret estimate
        # mean over K samples of phi(beta * <theta_star, theta_k>)
        proj = Theta_batch @ theta_star          # (K,)
        mean_reward_est = phi(proj, beta).mean()
        expected_regret[t] = (opt_reward - mean_reward_est)

    return expected_regret


def run_logistic_bandits_TS_exp(d, beta, T=200, num_exp=500, kappa=8.0, mh_steps=100, progress=False, chains=None):
    file_name = f"results_experiments/logistic_ts_beta_{beta}_d_{d}.pt"
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    if os.path.exists(file_name):
        print(f"Loading results from {file_name}...")
        expected_regret = torch.load(file_name, map_location=device)
    else:
        print(f"Running expensive experiments for beta={beta}, d={d}...")
        expected_regret = torch.zeros(T, 1, device=device)
        sampler = MHSphereSampler(d=d, kappa=kappa, device=device)

        # (Speed) skip tune_kappa; set a reasonable default and go.
        iterator = trange(num_exp, desc=f"β={beta}, d={d}", disable=not progress)
        for _ in iterator:
            expected_regret += bandit_experiment(d, beta, T, sampler, mh_steps=mh_steps, chains=chains)

        expected_regret = expected_regret / num_exp
        torch.save(expected_regret, file_name)
        print(f"Results saved to {file_name}.")
    return expected_regret


if __name__ == "__main__":
    d, beta, T = 10, 2.0, 200
    regs = run_logistic_bandits_TS_exp(d, beta, T=T, num_exp=100, kappa=8.0, mh_steps=100)
    plt.figure()
    plt.plot(regs.cpu().numpy(), label="TS (vMF–MH)")
    plt.xlabel("t")
    plt.ylabel("Expected regret")
    plt.legend()
    plt.show()
