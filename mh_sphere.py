import torch

class MHSphereSampler:
    """
    Exactly symmetric Metropolis–Hastings on S^{d-1}
    using von Mises–Fisher (vMF) random-walk proposals.
    Supports single-chain and batched multi-chain variants.
    """
    def __init__(self, d, kappa=8.0, device=None, dtype=torch.float32):
        self.d = d
        self.kappa = float(kappa)
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.dtype = dtype

    # ----------------- utilities -----------------
    def sample_uniform_sphere(self):
        x = torch.randn(self.d, device=self.device, dtype=self.dtype)
        return x / x.norm().clamp_min(1e-12)

    def sample_uniform_sphere_batch(self, K):
        x = torch.randn(K, self.d, device=self.device, dtype=self.dtype)
        return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def _householder_apply_batch(self, mu, x):
        K, d = mu.shape
        device, dtype = mu.device, mu.dtype
        e1 = torch.zeros(K, d, device=device, dtype=dtype)
        e1[:, 0] = 1.0
        v = e1 - mu                               # (K,d)
        nv = v.norm(dim=1, keepdim=True)
        v_unit = v / nv.clamp_min(1e-12)
        proj = (x * v_unit).sum(dim=1, keepdim=True)
        y = x - 2.0 * proj * v_unit
        mask = (nv.squeeze(1) <= 1e-12)
        if mask.any():
            y[mask] = x[mask]
        return y

    # ----------------- single-chain vMF -----------------
    def _vmf_sample(self, mu):
        d = mu.numel()
        device, dtype = mu.device, mu.dtype
        kappa_t = torch.as_tensor(self.kappa, device=device, dtype=dtype)
        if float(kappa_t) < 1e-8:
            x = torch.randn(d, device=device, dtype=dtype)
            return x / x.norm().clamp_min(1e-12)

        one = torch.ones((), device=device, dtype=dtype)
        dm1 = torch.as_tensor(d - 1, device=device, dtype=dtype)
        b = (-2.0 * kappa_t + torch.sqrt(4.0 * kappa_t * kappa_t + dm1 * dm1)) / dm1
        x = (one - b) / (one + b)
        c = kappa_t * x + dm1 * torch.log1p(-(x * x))

        beta_dist = torch.distributions.Beta(dm1 / 2.0, dm1 / 2.0)
        while True:
            z = beta_dist.sample()
            w = (one - (one + b) * z) / (one - (one - b) * z)
            u = torch.rand((), device=device, dtype=dtype)
            lhs = kappa_t * w + dm1 * torch.log1p(-(x * w))
            if (lhs - c) >= torch.log(u):
                break

        v = torch.randn(d - 1, device=device, dtype=dtype)
        v = v / v.norm().clamp_min(1e-12)
        theta_e1 = torch.cat((w.view(1), torch.sqrt(torch.clamp(one - w * w, min=0.0)) * v))
        e1 = torch.zeros(d, device=device, dtype=dtype); e1[0] = 1.0
        vH = e1 - mu
        nv = vH.norm()
        if nv <= 1e-12:
            return theta_e1
        vH = vH / nv
        return theta_e1 - 2.0 * (theta_e1 @ vH) * vH

    # ----------------- batched vMF -----------------
    def _vmf_sample_batch(self, mu):
        K, d = mu.shape
        device, dtype = mu.device, mu.dtype
        kappa_t = torch.as_tensor(self.kappa, device=device, dtype=dtype)

        if float(kappa_t) < 1e-8:
            x = torch.randn(K, d, device=device, dtype=dtype)
            return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)

        one = torch.ones((), device=device, dtype=dtype)
        dm1 = torch.as_tensor(d - 1, device=device, dtype=dtype)
        b = (-2.0 * kappa_t + torch.sqrt(4.0 * kappa_t * kappa_t + dm1 * dm1)) / dm1
        xparam = (one - b) / (one + b)
        c = kappa_t * xparam + dm1 * torch.log1p(-(xparam * xparam))

        beta_dist = torch.distributions.Beta(dm1 / 2.0, dm1 / 2.0)
        w = torch.empty(K, device=device, dtype=dtype)
        filled = torch.zeros(K, dtype=torch.bool, device=device)
        while not filled.all():
            need = (~filled).nonzero(as_tuple=False).squeeze(1)
            z = beta_dist.sample((need.numel(),)).to(device=device, dtype=dtype)
            w_try = (one - (one + b) * z) / (one - (one - b) * z)
            u = torch.rand(need.numel(), device=device, dtype=dtype)
            lhs = kappa_t * w_try + dm1 * torch.log1p(-(xparam * w_try))
            acc = (lhs - c) >= torch.log(u)
            if acc.any():
                idx = need[acc]
                w[idx] = w_try[acc]
                filled[idx] = True

        v = torch.randn(K, d - 1, device=device, dtype=dtype)
        v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
        sqrt_part = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0)).unsqueeze(1)
        theta_e1 = torch.cat([w.unsqueeze(1), sqrt_part * v], dim=1)
        theta = self._householder_apply_batch(mu, theta_e1)
        return theta / theta.norm(dim=1, keepdim=True).clamp_min(1e-12)

    # ----------------- MH steps -----------------
    def step(self, theta, logp_fn, n_steps=100):
        theta = theta.to(self.device, self.dtype)
        theta = theta / theta.norm().clamp_min(1e-12)
        logp = logp_fn(theta)
        for _ in range(n_steps):
            prop = self._vmf_sample(theta)
            logp_prop = logp_fn(prop)
            if torch.log(torch.rand((), device=self.device, dtype=self.dtype)) < (logp_prop - logp):
                theta, logp = prop, logp_prop
        return theta, logp

    def step_batch(self, theta_batch, logp_fn_batch, n_steps=50, return_accept_mask=False):
        """
        Batched MH for K chains.
        theta_batch: (K,d) unit vectors
        logp_fn_batch: callable((K,d)) -> (K,) log posterior
        If return_accept_mask=True, also returns a boolean mask of accepted proposals across all steps.
        """
        theta = theta_batch.to(self.device, self.dtype)
        theta = theta / theta.norm(dim=1, keepdim=True).clamp_min(1e-12)
        logp = logp_fn_batch(theta)  # (K,)
        K = theta.shape[0]
        any_accept = torch.zeros(K, dtype=torch.bool, device=self.device)
        for _ in range(n_steps):
            prop = self._vmf_sample_batch(theta)       # (K,d)
            logp_prop = logp_fn_batch(prop)            # (K,)
            u = torch.rand(K, device=self.device, dtype=self.dtype).log()
            accept = u < (logp_prop - logp)
            if accept.any():
                theta[accept] = prop[accept]
                logp[accept] = logp_prop[accept]
                any_accept |= accept
        if return_accept_mask:
            return theta, logp, any_accept
        return theta, logp
