import torch
from contextlib import nullcontext

class MHSphereSampler:
    """
    Metropolis–Hastings on S^{d-1} using vMF random-walk proposals.
      - States: Theta_bkd of shape (B, K, d)
      - logp_fn_group: callable((B, K, d)) -> (B, K) log posterior values
    """

    def __init__(self, d, kappa=8.0, device=None, dtype=torch.float32):
        self.d = int(d)
        self.kappa = float(kappa)
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.dtype = dtype

    # ---------- utilities ----------
    def sample_uniform_states(self, B: int, K: int):
        """
        Return (B, K, d) unit vectors uniformly on the sphere (for init).
        """
        x = torch.randn(B, K, self.d, device=self.device, dtype=self.dtype)
        return x / x.norm(dim=2, keepdim=True).clamp_min(1e-12)

    def _householder_apply_batch(self, mu, x):
        """
        Apply Householder that maps e1 -> mu row-wise.
        mu: (N, d) unit, x: (N, d) -> returns (N, d)
        """
        N, d = mu.shape
        device, dtype = mu.device, mu.dtype
        e1 = torch.zeros(N, d, device=device, dtype=dtype)
        e1[:, 0] = 1.0
        v = e1 - mu
        nv = v.norm(dim=1, keepdim=True)
        v_unit = v / nv.clamp_min(1e-12)
        proj = (x * v_unit).sum(dim=1, keepdim=True)
        y = x - 2.0 * proj * v_unit
        mask = (nv.squeeze(1) <= 1e-12)
        if mask.any():
            y[mask] = x[mask]
        return y

    # ---------- vMF sampler over (N, d) rows ----------
    def _vmf_sample_batch(self, mu_nd):
        """
        Draw one vMF proposal per row in mu_nd.
        mu_nd: (N, d) unit vectors (the current states)
        returns: (N, d) proposals
        """
        N, d = mu_nd.shape
        device, dtype = mu_nd.device, mu_nd.dtype
        kappa_t = torch.as_tensor(self.kappa, device=device, dtype=dtype)

        if float(kappa_t) < 1e-8:
            x = torch.randn(N, d, device=device, dtype=dtype)
            return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)

        one = torch.ones((), device=device, dtype=dtype)
        dm1 = torch.as_tensor(d - 1, device=device, dtype=dtype)
        b = (-2.0 * kappa_t + torch.sqrt(4.0 * kappa_t * kappa_t + dm1 * dm1)) / dm1
        xparam = (one - b) / (one + b)
        c = kappa_t * xparam + dm1 * torch.log1p(-(xparam * xparam))

        beta_dist = torch.distributions.Beta(dm1 / 2.0, dm1 / 2.0)
        w = torch.empty(N, device=device, dtype=dtype)
        filled = torch.zeros(N, dtype=torch.bool, device=device)
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

        v = torch.randn(N, d - 1, device=device, dtype=dtype)
        v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
        sqrt_part = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0)).unsqueeze(1)
        theta_e1 = torch.cat([w.unsqueeze(1), sqrt_part * v], dim=1)  # (N, d)
        theta = self._householder_apply_batch(mu_nd, theta_e1)
        return theta / theta.norm(dim=1, keepdim=True).clamp_min(1e-12)

    # ---------- MH over (B, K, d) ----------
    def _vmf_propose_bkd(self, Theta_bkd):
        """
        vMF RW proposals for each of the (B*K) current states.
        Theta_bkd: (B, K, d) -> returns (B, K, d)
        """
        B, K, d = Theta_bkd.shape
        flat = Theta_bkd.reshape(B * K, d)
        flat_prop = self._vmf_sample_batch(flat)
        return flat_prop.reshape(B, K, d)

    def mh_step(self, Theta_bkd, logp_fn_group, n_steps=12, use_amp=True):
        """
        MH for grouped (B experiments) × (K chains).
        Theta_bkd:      (B, K, d) current states (will be normalized)
        logp_fn_group:  callable((B, K, d)) -> (B, K) log posterior
        Returns: (Theta_bkd_new, logp_bk)
        """
        device, dtype = self.device, self.dtype
        autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if (use_amp and device.type == "cuda") else nullcontext()

        with autocast_ctx:
            Theta = Theta_bkd.to(device, dtype)
            Theta = Theta / Theta.norm(dim=2, keepdim=True).clamp_min(1e-12)
            logp  = logp_fn_group(Theta)  # (B, K)

            for _ in range(n_steps):
                prop = self._vmf_propose_bkd(Theta)      # (B, K, d)
                logp_prop = logp_fn_group(prop)          # (B, K)
                u = torch.rand_like(logp).log()
                accept = u < (logp_prop - logp)
                if accept.any():
                    Theta[accept] = prop[accept]
                    logp[accept]  = logp_prop[accept]
        return Theta, logp
