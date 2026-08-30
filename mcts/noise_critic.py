"""mcts/noise_critic.py

Noise-aware value model V(x, k) for classifier guidance (CG) on the frozen
Diffusion Forcing planner (mcts/df_model.py). This is the DUAL of the DF
planner: DFPlanner conditions the DENOISER on per-token noise levels k (B,T);
NoiseAwareCritic conditions the VALUE FUNCTION on the same per-token levels,
so it can score a partially-noised window x (B,T,D) — any mix of clean
history / denoising future / pure-noise tail the tree or the DF sampler can
produce — and return a predicted normalized return in [-1, 1].

Trajectory-level noise-aware classifier guidance is classical (Dhariwal &
Nichol 2021; Janner et al. 2022, Diffuser): train p(y|x_t) and steer sampling
with grad_x log p(y|x_t). The per-token generalization is the new bit here,
required because DF's noise level is per-token, not scalar — a single
trajectory-level t has no meaning once history is clean and the future is
still noisy mid-sweep. The critic is trained on the MIXED level distribution
mcts.df_schedule.sample_training_levels produces: uniform coverage (so V is
defined everywhere) blended with pyramid-plus-clean-prefix rows (the EXACT
per-token pattern DFPlanner.sample walks through, prefixed by the h clean
history tokens tree expansion conditions on) — that second component is the
query distribution guidance is actually evaluated on at inference, so
training must cover it directly rather than relying on generalization from
uniform noise alone.

Two consumers:
  * eps-shift classifier guidance inside DFPlanner.sample (mcts/df_model.py):
    eps <- eps - w_cg * sqrt(1 - alpha_bar[k]) * grad_x V(x, k), i.e.
    conditional score steering, self-annealing per token as it denoises;
  * a potential in-tree leaf evaluator (mcts/mcts_loop.py, out of scope here)
    that can score a node's partially-denoised window directly instead of
    waiting for a fully clean sample.

Architecture: CausalDFDiT's per-token-adaLN transformer (mcts/df_model.py),
made BIDIRECTIONAL (no causal mask — a value estimate may legitimately use
future tokens, since guidance is applied to a whole window at once, not
autoregressively) and terminated in mean-pool + scalar head instead of a
per-token eps head. Blocks are reused (CausalDFBlock with causal_mask=None),
not reimplemented.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cleandiffuser.utils import SinusoidalEmbedding
from mcts.df_model import CausalDFBlock, _modulate
from mcts.df_schedule import alpha_bar_cosine


class NoiseCriticNet(nn.Module):
    """V(x, k) net: (x (B,T,D), k (B,T) int levels) -> value (B,)."""

    def __init__(self, in_dim: int, d_model: int = 256, n_heads: int = 4,
                 depth: int = 2, emb_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.in_dim, self.d_model = in_dim, d_model
        self.x_proj = nn.Linear(in_dim, d_model)
        self.k_emb = SinusoidalEmbedding(emb_dim)
        self.map_emb = nn.Sequential(
            nn.Linear(emb_dim, d_model), nn.Mish(),
            nn.Linear(d_model, d_model), nn.Mish())
        self.pos_emb = SinusoidalEmbedding(d_model)
        self.blocks = nn.ModuleList(
            [CausalDFBlock(d_model, n_heads, dropout) for _ in range(depth)])
        self.norm_f = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.adaLN_f = nn.Sequential(nn.SiLU(), nn.Linear(d_model, d_model * 2))
        self.head = nn.Linear(d_model, 1)
        self._init_weights()

    def _init_weights(self):
        def basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(basic)
        for blk in self.blocks:                      # adaLN-zero init (DiT)
            nn.init.constant_(blk.adaLN[-1].weight, 0)
            nn.init.constant_(blk.adaLN[-1].bias, 0)
        nn.init.constant_(self.adaLN_f[-1].weight, 0)
        nn.init.constant_(self.adaLN_f[-1].bias, 0)
        # zero-init the scalar head: V starts at 0 everywhere, so the
        # guidance gradient starts null and training starts unbiased.
        nn.init.constant_(self.head.weight, 0)
        nn.init.constant_(self.head.bias, 0)

    def forward(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        pos = self.pos_emb(torch.arange(T, device=x.device))       # (T, d)
        h = self.x_proj(x) + pos[None]
        ke = self.k_emb(k.reshape(-1).float()).reshape(B, T, -1)   # (B, T, e)
        emb = self.map_emb(ke)                                     # (B, T, d)
        for blk in self.blocks:
            h = blk(h, emb, None)      # causal_mask=None -> bidirectional attn
        sf, cf = self.adaLN_f(emb).chunk(2, dim=-1)
        h = _modulate(self.norm_f(h), sf, cf)
        pooled = h.mean(dim=1)                                     # (B, d)
        return self.head(pooled).squeeze(-1)                       # (B,)


class NoiseAwareCritic:
    """Training + inference wrapper (net, EMA, alpha-bar table, save/load) —
    mirrors DFPlanner's conventions exactly so the two can share a training
    loop shape and a checkpoint-loading dispatch pattern.
    """

    def __init__(self, in_dim: int, K: int = 20, d_model: int = 256,
                 n_heads: int = 4, depth: int = 2, emb_dim: int = 128,
                 dropout: float = 0.0, ema_rate: float = 0.999,
                 device: str = "cpu"):
        self.cfg = dict(kind="noise_critic", in_dim=in_dim, K=K,
                        d_model=d_model, n_heads=n_heads, depth=depth,
                        emb_dim=emb_dim, dropout=dropout, ema_rate=ema_rate)
        self.K, self.dev = K, device
        self.ema_rate = ema_rate
        self.net = NoiseCriticNet(in_dim, d_model, n_heads, depth, emb_dim,
                                  dropout).to(device)
        self.net_ema = NoiseCriticNet(in_dim, d_model, n_heads, depth, emb_dim,
                                      dropout).to(device)
        self.net_ema.load_state_dict(self.net.state_dict())
        self.net_ema.requires_grad_(False).eval()
        ab = torch.tensor(alpha_bar_cosine(K), device=device)
        self.sqrt_ab = ab.sqrt()                       # (K+1,)
        self.sqrt_1mab = (1.0 - ab).clamp_min(0.0).sqrt()

    def loss(self, x0: torch.Tensor, val: torch.Tensor,
             k: torch.Tensor) -> torch.Tensor:
        """x0: (B,T,D) clean window, val: (B,) normalized-return label in
        [-1,1], k: (B,T) int noise levels (typically drawn from
        mcts.df_schedule.sample_training_levels). Noises x0 to level k and
        regresses the return.

        Unlike DFPlanner.loss, k=0 (clean) tokens ARE fully trainable here:
        the eps-loss is unidentifiable at k=0 (there is no noise to predict),
        but the value-regression loss has no such degeneracy — a clean
        window still has a well-defined return target, so every level
        including 0 contributes to the loss.
        """
        eps = torch.randn_like(x0)
        xk = (self.sqrt_ab[k].unsqueeze(-1) * x0
              + self.sqrt_1mab[k].unsqueeze(-1) * eps)
        pred = self.net(xk, k)
        return F.mse_loss(pred, val)

    def value(self, x: torch.Tensor, k: torch.Tensor,
              use_ema: bool = True) -> torch.Tensor:
        """(B,) predicted return. Deliberately NOT @torch.no_grad()-wrapped:
        classifier guidance needs grad_x of this w.r.t. its input."""
        net = self.net_ema if use_ema else self.net
        return net(x, k)

    def grad_value(self, x: torch.Tensor, k: torch.Tensor,
                   use_ema: bool = True) -> torch.Tensor:
        """(B,T,D) grad_x V(x,k).sum(). Convenience wrapper for guidance
        call sites that may run inside torch.no_grad() (e.g. DFPlanner.sample
        is @torch.no_grad()-decorated) — torch.enable_grad() locally
        overrides that so the gradient can still be taken."""
        with torch.enable_grad():
            xg = x.detach().requires_grad_(True)
            v = self.value(xg, k, use_ema=use_ema)
            g = torch.autograd.grad(v.sum(), xg)[0]
        return g

    @torch.no_grad()
    def ema_update(self):
        r = self.ema_rate
        for pe, p in zip(self.net_ema.parameters(), self.net.parameters()):
            pe.mul_(r).add_(p, alpha=1.0 - r)
        for be, b in zip(self.net_ema.buffers(), self.net.buffers()):
            be.copy_(b)

    # ── persistence ─────────────────────────────────────────────────────────
    def save(self, path: str, **meta):
        torch.save(dict(net=self.net.state_dict(),
                        net_ema=self.net_ema.state_dict(),
                        cfg=self.cfg, meta=meta), path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "NoiseAwareCritic":
        d = torch.load(path, map_location=device, weights_only=False)
        cfg = dict(d["cfg"])
        kind = cfg.pop("kind", "noise_critic")   # __init__ doesn't take "kind"
        c = cls(device=device, **cfg)
        c.cfg["kind"] = kind
        c.net.load_state_dict(d["net"])
        c.net_ema.load_state_dict(d["net_ema"])
        c.net.eval()
        c.meta = d.get("meta", {})
        return c
