"""mcts/value_net.py

State-value critic V(s) for MCTS-as-sampler over Diffusion Veteran.

Why a new critic
----------------
DV's MCSS critic (DVHorizonCritic) scores a FULL (H, D) trajectory -> one scalar
return-to-go.  That value does full lookahead at ply 1, so a tree built on it gains
nothing from depth: every node re-evaluates the whole remaining return (this is exactly
why the Phase-4 measurements showed MCTS == MCSS at matched K).  For MCTS to *compose*
segments across depth, the value must be a function of a single STATE:

    V(s)  ->  normalised return-to-go from s

trained on the *same* target the MCSS critic regresses, just keyed per-state instead of
per-trajectory.  Concretely the supervision is the pair

    ( seq_obs[p, t] ,  seq_val[p, t] )           for valid start indices (p, t)

drawn from the same DV dataset (dataset.indices).  seq_obs is already
GaussianNormalizer-normalised (identical to the `s_norm` the tree carries at inference),
and seq_val is the normalised discounted return-to-go.  Restricting to dataset.indices
gives the identical supervision distribution to the MCSS critic, so the two are directly
comparable: only the input representation (single state vs whole trajectory) changes.

The net is a small MLP.  A per-state value needs no sequence model, and an MLP is cheap
enough to score every tree node without dominating the planner's diffusion cost.

This module deliberately lives under mcts/ (not cleandiffuser/) so the new critic is
fully isolated from the DV training pipeline and the cleandiffuser package is untouched.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class DVStateValue(nn.Module):
    """MLP state-value head:  V(s) -> scalar.

    Operates on the last dimension only, so it accepts any leading batch shape:
        (obs_dim,)        -> (1,)
        (B, obs_dim)      -> (B, 1)
        (N, K, obs_dim)   -> (N, K, 1)

    Args:
        obs_dim:    observation dimension (4 for maze2d, 29 for antmaze).
        hidden_dim: width of each hidden layer.
        depth:      number of hidden layers (>= 1).
        dropout:    optional dropout after each hidden activation.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 256,
        depth: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.dropout = dropout

        layers: list[nn.Module] = [
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        ]
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """s: (..., obs_dim) -> (..., 1)."""
        return self.net(s)

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str, **extra) -> None:
        """Save weights plus the architecture hyper-params (so load() needs no guessing)."""
        torch.save(
            {
                "state_value": self.state_dict(),
                "obs_dim": self.obs_dim,
                "hidden_dim": self.hidden_dim,
                "depth": self.depth,
                "dropout": self.dropout,
                **extra,
            },
            path,
        )


def load_state_value(
    path: str,
    device: str = "cpu",
    obs_dim: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    depth: Optional[int] = None,
    dropout: Optional[float] = None,
) -> DVStateValue:
    """Load a DVStateValue from a checkpoint saved by DVStateValue.save().

    Architecture hyper-params are read from the checkpoint when present; explicit
    arguments override them (useful for older checkpoints that lack the metadata).
    `dropout` must match the trained value: it changes how many nn.Dropout modules sit
    in the Sequential, which shifts the parameter indices in the state_dict.  Checkpoints
    that predate the metadata fall back to dropout=0.0 (the training default).
    The returned net is in eval mode.

    weights_only=False is explicit: the checkpoint stores non-tensor metadata (obs_dim,
    env name, center_mapping), which the torch>=2.6 weights_only=True default would reject.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    ckpt_obs_dim = ckpt.get("obs_dim")
    if obs_dim is None and ckpt_obs_dim is None:
        raise ValueError(
            "obs_dim not found in checkpoint and not supplied — pass obs_dim explicitly")
    obs_dim = obs_dim if obs_dim is not None else ckpt_obs_dim
    hidden_dim = hidden_dim if hidden_dim is not None else ckpt.get("hidden_dim", 256)
    depth = depth if depth is not None else ckpt.get("depth", 3)
    dropout = dropout if dropout is not None else ckpt.get("dropout", 0.0)
    net = DVStateValue(obs_dim, hidden_dim=hidden_dim, depth=depth,
                       dropout=dropout).to(device)
    net.load_state_dict(ckpt["state_value"])
    net.eval()
    return net
