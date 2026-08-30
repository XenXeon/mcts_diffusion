"""mcts/shortcut_df.py

Shortcut-forcing planner: Diffusion Forcing's per-token noise + Shortcut
Models' few-step sampling — the Dreamer 4 world-model recipe, built on this
repo's CausalDFDiT backbone.

Shortcut models [Frans, Hafner, Levine, Abbeel, 2024, arXiv:2410.12557]
condition the denoiser on the STEP SIZE d as well as the time t, so it learns
the correct big-jump integrator instead of the instantaneous velocity (naive
big Euler steps on a curved ODE average over modes and fail). Training is
end-to-end: a flow-matching base case at d=0 grounds small steps; a
self-consistency bootstrap  s(x,t,2d) = [s(x,t,d) + s(x',t+d,d)] / 2  (targets
from the EMA net, stop-gradient) propagates quality to large steps. Paper
recipe followed: velocity parameterization x_t = (1-t)·noise + t·data, dyadic
step grid (base_units=128 -> d in {1/128..1}), ~1/4 of each batch on the
bootstrap term, t sampled on multiples of d for bootstrap rows, d=0 queries
for the smallest half-step, EMA targets, weight decay 0.1 (paper: crucial).

WHY here: the DF pyramid's 52 sweeps are dominated by the causal-lag term
(T-1=31), not the 20 noise levels — few-step sampling only pays if the free
tokens step JOINTLY (fullseq-style), which this class does in `sweeps` (default
4) model forwards vs the DF planner's 52: ~13x faster tree expansion. Whether
dropping the "far-future-noisier" pyramid costs plan quality is an empirical
gate: compare open-loop critic scores (scripts/check_df_ckpt.py) and DF-MCSS
closed-loop against the standard DF planner BEFORE trusting tree arms.

Diffusion-forcing property kept: t is PER-TOKEN in training (clean-context
tokens included), so clean-history + noisy-future inputs stay in-distribution
and tree expansion remains exact conditional generation.

Train:  scripts/train_df_planner.py --env maze2d-large-v1 --shortcut --out-tag shortcut
Deploy: identical to the DF planner (--df-ckpt shortcut) — load_df_planner
dispatches on the saved cfg; .sample() ignores pyramid-only kwargs.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from mcts.df_model import CausalDFDiT, _modulate


class ShortcutDFDiT(CausalDFDiT):
    """CausalDFDiT with (t, d) conditioning: cond is (B, T, 2), embeddings of
    the two scaled scalars are summed (SinusoidalEmbedding is parameter-free,
    so the module tree — and therefore checkpoint layout — is unchanged)."""

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        pos = self.pos_emb(torch.arange(T, device=x.device))
        h = self.x_proj(x) + pos[None]
        ke = (self.k_emb(cond[..., 0].reshape(-1).float()).reshape(B, T, -1)
              + self.k_emb(cond[..., 1].reshape(-1).float()).reshape(B, T, -1))
        emb = self.map_emb(ke)
        mask = self._causal_mask(T, x.device)
        for blk in self.blocks:
            h = blk(h, emb, mask)
        sf, cf = self.adaLN_f(emb).chunk(2, dim=-1)
        return self.out(_modulate(self.norm_f(h), sf, cf))


class ShortcutDFPlanner:
    """Training + few-step sampling wrapper. Same interface as DFPlanner."""

    def __init__(self, in_dim: int, base_units: int = 128, d_model: int = 256,
                 n_heads: int = 4, depth: int = 4, emb_dim: int = 128,
                 dropout: float = 0.0, ema_rate: float = 0.999,
                 default_sweeps: int = 4, boot_frac: float = 0.25,
                 clean_frac: float = 0.125, x_clip: float = 8.0,
                 device: str = "cpu", kind: str = "shortcut"):
        assert kind == "shortcut"
        self.cfg = dict(in_dim=in_dim, base_units=base_units, d_model=d_model,
                        n_heads=n_heads, depth=depth, emb_dim=emb_dim,
                        dropout=dropout, ema_rate=ema_rate,
                        default_sweeps=default_sweeps, boot_frac=boot_frac,
                        clean_frac=clean_frac, x_clip=x_clip, kind="shortcut")
        self.base_units, self.dev = base_units, device
        self.n_dyadic = int(math.log2(base_units))     # j in {1..n}: d = 2^j/M
        assert 2 ** self.n_dyadic == base_units, "base_units must be a power of 2"
        self.default_sweeps, self.boot_frac = default_sweeps, boot_frac
        self.clean_frac, self.x_clip = clean_frac, x_clip
        self.ema_rate = ema_rate
        self.net = ShortcutDFDiT(in_dim, d_model, n_heads, depth, emb_dim,
                                 dropout).to(device)
        self.net_ema = ShortcutDFDiT(in_dim, d_model, n_heads, depth, emb_dim,
                                     dropout).to(device)
        self.net_ema.load_state_dict(self.net.state_dict())
        self.net_ema.requires_grad_(False).eval()

    # ── training loss (paper Algorithm 1, per-token t = diffusion forcing) ──
    def loss(self, x1: torch.Tensor) -> torch.Tensor:
        """x1: (B, T, D) clean data windows. Returns combined shortcut loss."""
        B, T, D = x1.shape
        dev = x1.device
        M = float(self.base_units)
        n_boot = max(1, int(round(B * self.boot_frac)))
        n_emp = B - n_boot
        x0 = torch.randn_like(x1)                       # x0 = noise, x1 = data

        # empirical flow-matching at d=0; per-token t ~ U[0,1) plus explicit
        # clean tokens (t=1) so the clean-history context pathway is trained
        te = torch.rand(n_emp, T, device=dev)
        te = torch.where(torch.rand(n_emp, T, device=dev) < self.clean_frac,
                         torch.ones_like(te), te)
        xe = (1 - te)[..., None] * x0[:n_emp] + te[..., None] * x1[:n_emp]
        cond_e = torch.stack([te * M, torch.zeros_like(te)], dim=-1)
        l_emp = F.mse_loss(self.net(xe, cond_e), x1[:n_emp] - x0[:n_emp])

        # self-consistency: per-sequence dyadic d_target = 2^j / M (j >= 1),
        # per-token t on multiples of d_target in [0, 1 - d_target]
        xb0, xb1 = x0[n_emp:], x1[n_emp:]
        j = torch.randint(1, self.n_dyadic + 1, (n_boot,), device=dev)
        d2 = (2.0 ** j.float()) / M                     # (n_boot,) trained step
        dh = d2 / 2.0                                   # executed half-step
        cells = (M / (2.0 ** j.float()))                # = 1/d2, integer-valued
        idx = torch.floor(torch.rand(n_boot, T, device=dev) * cells[:, None])
        idx = torch.minimum(idx, cells[:, None] - 1.0)
        tb = idx * d2[:, None]
        xb = (1 - tb)[..., None] * xb0 + tb[..., None] * xb1
        # paper: when the half-step is the smallest unit, query the net at d=0
        dh_cond = torch.where(j == 1, torch.zeros_like(dh), dh)
        with torch.no_grad():                           # EMA bootstrap targets
            c1 = torch.stack([tb * M, (dh_cond[:, None] * M).expand_as(tb)], -1)
            s1 = self.net_ema(xb, c1)
            xh = xb + s1 * dh[:, None, None]
            c2 = torch.stack([(tb + dh[:, None]) * M,
                              (dh_cond[:, None] * M).expand_as(tb)], -1)
            s2 = self.net_ema(xh, c2)
            target = ((s1 + s2) / 2.0).detach()
        cb = torch.stack([tb * M, (d2[:, None] * M).expand_as(tb)], -1)
        l_boot = F.mse_loss(self.net(xb, cb), target)
        return l_emp + l_boot

    @torch.no_grad()
    def ema_update(self):
        r = self.ema_rate
        for pe, p in zip(self.net_ema.parameters(), self.net.parameters()):
            pe.mul_(r).add_(p, alpha=1.0 - r)
        for be, b in zip(self.net_ema.buffers(), self.net.buffers()):
            be.copy_(b)

    # ── few-step sampling: history tokens clean, free tokens step JOINTLY ──
    @torch.no_grad()
    def sample(self, x_hist: torch.Tensor, hist_len: torch.Tensor, T: int,
               sweeps: int = None, temperature: float = 1.0,
               use_ema: bool = True, schedule: str = None, slope: int = None,
               row_stride: int = None, **_) -> torch.Tensor:
        """Same contract as DFPlanner.sample (rows [0:hist_len[i]] = history,
        returned untouched). Pyramid-only kwargs (schedule/slope/row_stride)
        are accepted and ignored so the Sampler is backbone-agnostic. `sweeps`
        must be a power of two <= base_units; each sweep is ONE net forward.
        """
        net = self.net_ema if use_ema else self.net
        sweeps = int(sweeps or self.default_sweeps)
        if sweeps < 1 or (sweeps & (sweeps - 1)) or sweeps > self.base_units:
            raise ValueError(f"sweeps must be a power of 2 in [1, "
                             f"{self.base_units}], got {sweeps}")
        n, D = x_hist.shape[0], x_hist.shape[2]
        M = float(self.base_units)
        cols = torch.arange(T, device=self.dev)
        hist = cols[None, :] < hist_len[:, None].to(self.dev)      # (n, T)
        d = 1.0 / sweeps
        d_cond = 0.0 if sweeps == self.base_units else d           # paper rule
        x = torch.randn(n, T, D, device=self.dev) * temperature
        x = torch.where(hist.unsqueeze(-1), x_hist, x)
        t_free = 0.0
        for _step in range(sweeps):
            t_tok = torch.where(hist, torch.ones(n, T, device=self.dev),
                                torch.full((n, T), t_free, device=self.dev))
            d_tok = torch.where(hist, torch.zeros(n, T, device=self.dev),
                                torch.full((n, T), d_cond, device=self.dev))
            s = net(x, torch.stack([t_tok * M, d_tok * M], dim=-1))
            x = torch.where(hist.unsqueeze(-1), x, x + s * d)
            x = x.clamp(-self.x_clip, self.x_clip)
            x = torch.where(hist.unsqueeze(-1), x_hist, x)         # exact
            t_free += d
        return x

    # ── persistence (same file scheme as DFPlanner; kind dispatches) ──
    def save(self, path: str, **meta):
        torch.save(dict(net=self.net.state_dict(),
                        net_ema=self.net_ema.state_dict(),
                        cfg=self.cfg, meta=meta), path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "ShortcutDFPlanner":
        d = torch.load(path, map_location=device, weights_only=False)
        p = cls(device=device, **d["cfg"])
        p.net.load_state_dict(d["net"])
        p.net_ema.load_state_dict(d["net_ema"])
        p.net.eval()
        p.meta = d.get("meta", {})
        return p
