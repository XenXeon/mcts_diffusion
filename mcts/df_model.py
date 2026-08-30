"""mcts/df_model.py

Causal Diffusion Forcing planner — a faithful minimal implementation of
Diffusion Forcing (Chen et al., NeurIPS 2024, arXiv 2407.01392) Algorithms
1-2, as the causal-transformer variant their Appendix B.1 sketches, built to
slot into THIS repo's DV harness:

  * tokens = stride-spaced waypoint rows, the SAME (H, obs_dim) normalized
    windows the DV planner trains on (dataset __getitem__ distribution), so
    the DV trajectory critic can score DF windows and the DV inverse-dynamics
    policy can execute them — planner is the ONLY component swapped;
  * NO classifier guidance: selection stays sample-and-rank with the DV
    critic (DV's own study: rank > guidance), which also keeps the DF arm's
    evaluator identical to every other arm in the study.

Why this exists (the inpaint post-mortem, notes/value_lever_findings.md §5b):
prefix-inpainting on the frozen DV planner feeds it clean-prefix + noisy-
future inputs it never saw (trained with row-0 clamping only) — measured
-16 points closed-loop. Diffusion Forcing trains on independent per-token
noise levels, so "clean history + noisy future" is IN-distribution by
construction: conditioning a continuation on a search prefix is exact
conditional sampling, not replacement approximation.

Architecture: DiT1d's blocks (cleandiffuser/nn_diffusion/dit.py) with two
changes — a causal attention mask, and per-token adaLN modulation driven by
a per-token noise-level embedding (noise levels are (B, T) ints, 0 = clean).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cleandiffuser.utils import SinusoidalEmbedding
from mcts.df_schedule import alpha_bar_cosine, fullseq_matrix, pyramid_matrix


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1.0 + scale) + shift          # all (B, T, d) — per-token adaLN


class CausalDFBlock(nn.Module):
    """DiTBlock with causal attention and per-token adaLN conditioning."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(approximate="tanh"),
            nn.Dropout(dropout), nn.Linear(d_model * 4, d_model))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_model, d_model * 6))

    def forward(self, x: torch.Tensor, emb: torch.Tensor,
                causal_mask: torch.Tensor):
        # emb: (B, T, d) per-token — chunk along the LAST dim (vs DiT's (B, d))
        sm, cm, gm, sp, cp, gp = self.adaLN(emb).chunk(6, dim=-1)
        h = _modulate(self.norm1(x), sm, cm)
        x = x + gm * self.attn(h, h, h, attn_mask=causal_mask,
                               need_weights=False)[0]
        x = x + gp * self.mlp(_modulate(self.norm2(x), sp, cp))
        return x


class CausalDFDiT(nn.Module):
    """eps prediction net: (x (B,T,D), k (B,T) int levels) -> eps (B,T,D)."""

    def __init__(self, in_dim: int, d_model: int = 256, n_heads: int = 4,
                 depth: int = 4, emb_dim: int = 128, dropout: float = 0.0):
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
        self.out = nn.Linear(d_model, in_dim)
        self._mask_cache: Optional[torch.Tensor] = None
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
        nn.init.constant_(self.out.weight, 0)
        nn.init.constant_(self.out.bias, 0)

    def _causal_mask(self, T: int, device) -> torch.Tensor:
        m = self._mask_cache
        if m is None or m.shape[0] != T or m.device != device:
            m = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool),
                           diagonal=1)          # True = NOT allowed to attend
            self._mask_cache = m
        return m

    def forward(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        pos = self.pos_emb(torch.arange(T, device=x.device))       # (T, d)
        h = self.x_proj(x) + pos[None]
        ke = self.k_emb(k.reshape(-1).float()).reshape(B, T, -1)   # (B, T, e)
        emb = self.map_emb(ke)                                     # (B, T, d)
        mask = self._causal_mask(T, x.device)
        for blk in self.blocks:
            h = blk(h, emb, mask)
        sf, cf = self.adaLN_f(emb).chunk(2, dim=-1)
        return self.out(_modulate(self.norm_f(h), sf, cf))


class DFPlanner:
    """Training + sampling wrapper (net, EMA, alpha-bar table, save/load).

    Level convention: k in {0..K}, alpha_bar[0] = 1 (clean), alpha_bar[K] ~ 0.
    """

    def __init__(self, in_dim: int, K: int = 20, d_model: int = 256,
                 n_heads: int = 4, depth: int = 4, emb_dim: int = 128,
                 dropout: float = 0.0, ema_rate: float = 0.999,
                 x0_clip: float = 6.0, device: str = "cpu"):
        self.cfg = dict(in_dim=in_dim, K=K, d_model=d_model, n_heads=n_heads,
                        depth=depth, emb_dim=emb_dim, dropout=dropout,
                        ema_rate=ema_rate, x0_clip=x0_clip)
        self.K, self.dev, self.x0_clip = K, device, x0_clip
        self.ema_rate = ema_rate
        self.net = CausalDFDiT(in_dim, d_model, n_heads, depth, emb_dim,
                               dropout).to(device)
        self.net_ema = CausalDFDiT(in_dim, d_model, n_heads, depth, emb_dim,
                                   dropout).to(device)
        self.net_ema.load_state_dict(self.net.state_dict())
        self.net_ema.requires_grad_(False).eval()
        ab = torch.tensor(alpha_bar_cosine(K), device=device)
        self.sqrt_ab = ab.sqrt()                       # (K+1,)
        self.sqrt_1mab = (1.0 - ab).clamp_min(0.0).sqrt()

    # ── Algorithm 1: independent per-token noise levels ─────────────────────
    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        B, T, _ = x0.shape
        k = torch.randint(0, self.K + 1, (B, T), device=x0.device)
        eps = torch.randn_like(x0)
        xk = (self.sqrt_ab[k].unsqueeze(-1) * x0
              + self.sqrt_1mab[k].unsqueeze(-1) * eps)
        pred = self.net(xk, k)
        # k=0 tokens are exactly clean: eps is unidentifiable from the input,
        # so they train the CONTEXT pathway (clean history among noisy tokens
        # — the configuration tree search conditions on) but not the loss.
        w = (k > 0).float().unsqueeze(-1)
        return (w * (pred - eps) ** 2).sum() / w.sum().clamp_min(1.0)

    @torch.no_grad()
    def ema_update(self):
        r = self.ema_rate
        for pe, p in zip(self.net_ema.parameters(), self.net.parameters()):
            pe.mul_(r).add_(p, alpha=1.0 - r)
        for be, b in zip(self.net_ema.buffers(), self.net.buffers()):
            be.copy_(b)

    # ── Algorithm 2: matrix-schedule sampling with native history tokens ────
    @torch.no_grad()
    def sample(self, x_hist: torch.Tensor, hist_len: torch.Tensor, T: int,
               schedule: str = "pyramid", slope: int = 1, row_stride: int = 1,
               temperature: float = 1.0, use_ema: bool = True,
               guide=None, w_cg: float = 0.0, **_) -> torch.Tensor:
        """Generate (n, T, in_dim) windows whose first hist_len[i] tokens are
        the given clean history — EXACT conditional sampling (history tokens
        ride at level 0 the whole way; the causal net saw clean-past + noisy-
        future inputs throughout training).

        x_hist: (n, T, D) with rows [0:hist_len[i]] = history (rest ignored),
        hist_len: (n,) ints >= 1 (row 0 = current state at minimum).

        guide: optional NoiseAwareCritic (mcts/noise_critic.py). When given
        (with w_cg != 0), each step applies the classifier-guidance eps-shift
        eps <- eps - w_cg * sqrt(1 - alpha_bar[k]) * grad_x V(x, k) — Diffuser-
        style conditional score steering, generalized to per-token k. The
        sqrt(1-alpha_bar[k]) factor is per-token, so guidance self-anneals to
        zero as each individual token approaches clean (k -> 0), matching the
        per-token noise schedule rather than a single trajectory-level t.
        Default guide=None reproduces every existing call site bit-identically.
        """
        net = self.net_ema if use_ema else self.net
        n, D = x_hist.shape[0], x_hist.shape[2]
        mat = (pyramid_matrix(self.K, T, slope, row_stride)
               if schedule == "pyramid" else
               fullseq_matrix(self.K, T, row_stride))
        mat = torch.as_tensor(mat, device=self.dev)                 # (M+1, T)
        cols = torch.arange(T, device=self.dev)
        hist_mask = cols[None, :] < hist_len[:, None].to(self.dev)  # (n, T)

        x = torch.randn(n, T, D, device=self.dev) * temperature
        x = torch.where(hist_mask.unsqueeze(-1), x_hist, x)
        k_prev = mat[0][None].expand(n, T).clone()
        k_prev = torch.where(hist_mask, torch.zeros_like(k_prev), k_prev)
        for m in range(1, mat.shape[0]):
            k_new = mat[m][None].expand(n, T).clone()
            k_new = torch.where(hist_mask, torch.zeros_like(k_new), k_new)
            sa_p = self.sqrt_ab[k_prev].unsqueeze(-1)
            s1_p = self.sqrt_1mab[k_prev].unsqueeze(-1)
            eps = net(x, k_prev)
            if guide is not None and w_cg:
                with torch.enable_grad():
                    xg = x.detach().requires_grad_(True)
                    g = torch.autograd.grad(guide.value(xg, k_prev).sum(), xg)[0]
                # zero out history columns: they are clamped back to x_hist
                # below regardless, but the eps shift must not leak guidance
                # into the x0 estimate of already-clean history tokens.
                g = torch.where(hist_mask.unsqueeze(-1), torch.zeros_like(g), g)
                eps = eps - w_cg * s1_p * g
            upd = (k_new < k_prev).unsqueeze(-1)
            x0 = ((x - s1_p * eps) / sa_p.clamp_min(1e-4)).clamp(
                -self.x0_clip, self.x0_clip)
            x_next = (self.sqrt_ab[k_new].unsqueeze(-1) * x0
                      + self.sqrt_1mab[k_new].unsqueeze(-1) * eps)  # DDIM
            x = torch.where(upd, x_next, x)
            x = torch.where(hist_mask.unsqueeze(-1), x_hist, x)     # exact
            k_prev = k_new
        return x

    # ── persistence ─────────────────────────────────────────────────────────
    def save(self, path: str, **meta):
        torch.save(dict(net=self.net.state_dict(),
                        net_ema=self.net_ema.state_dict(),
                        cfg=self.cfg, meta=meta), path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "DFPlanner":
        d = torch.load(path, map_location=device, weights_only=False)
        p = cls(device=device, **d["cfg"])
        p.net.load_state_dict(d["net"])
        p.net_ema.load_state_dict(d["net_ema"])
        p.net.eval()
        p.meta = d.get("meta", {})
        return p


def load_df_planner(path: str, device: str = "cpu"):
    """Dispatching loader: DFPlanner or ShortcutDFPlanner by the saved cfg.

    All df_planner_ckpt_*.pt files share one naming scheme; the cfg's "kind"
    key ("shortcut" = mcts/shortcut_df.py few-step model, absent = the standard
    DF planner) decides the class. Both classes expose the same .sample()
    signature (extra kwargs ignored), so the harness is backbone-agnostic.
    """
    d = torch.load(path, map_location="cpu", weights_only=False)
    if d.get("cfg", {}).get("kind") == "shortcut":
        from mcts.shortcut_df import ShortcutDFPlanner
        return ShortcutDFPlanner.load(path, device=device)
    return DFPlanner.load(path, device=device)
