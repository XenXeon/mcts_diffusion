"""mcts/window.py

Torch-free helpers for CRITIC-mode tree search: keep every node's value on ONE
comparable scoring window.

The DV trajectory critic scores an H-step window and is trained on windows that
start at the state the plan was sampled from. A naive critic-in-tree scores each
node's continuation on the continuation's OWN window [t_node, t_node+H), which
shifts forward with depth; on progress/camping tasks the later window scores
systematically higher (more of it is spent near the goal), so max-backup inflates
whichever children the search happened to expand — a visit-count bias, not merit.

The fix implemented here: every node is scored on the SAME window [s0, s0+H)
anchored at the current REAL state. The search-chosen prefix of waypoints is
prepended to the planner's continuation and the first H waypoints of that
composite are scored. Backed-up values then compare like with like, and the tree
is a best-first search over trajectory PREFIXES, all judged by the same
well-posed critic that MCSS uses.

Prefix convention: a node's prefix holds the waypoint rows from s0 up to but NOT
including the node's own state (root prefix = None). A continuation sampled at
the node has the node's state inpainted at row 0, so prefix + continuation is
contiguous and stride-spaced — exactly the object the critic was trained on.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def compose_window(prefix: Optional[np.ndarray], trajs: np.ndarray) -> np.ndarray:
    """(d, D) prefix + (K, H, D) continuations -> (K, H, D) windows starting at s0.

    Output row k is the first H waypoints of prefix ++ trajs[k]. prefix=None or
    length 0 is the root case: continuations already start at s0, returned as-is.
    Raises if the prefix alone fills the window (the search is deeper than the
    critic can see — keep budget * child_index <= H - 1).
    """
    K, H, D = trajs.shape
    d = 0 if prefix is None else int(prefix.shape[0])
    if d == 0:
        return trajs
    if d >= H:
        raise ValueError(
            f"prefix length {d} >= horizon {H}: node is deeper than the critic "
            f"window; keep budget * child_index <= H - 1")
    pref = np.broadcast_to(prefix, (K, d, D))
    return np.concatenate([pref, trajs[:, :H - d]], axis=1)


def extend_prefix(prefix: Optional[np.ndarray], traj: np.ndarray,
                  child_index: int) -> np.ndarray:
    """Prefix for the child reached via `traj` (one (H, D) continuation).

    The child's state is traj[child_index], so its prefix gains rows
    traj[0:child_index] — the parent node's state plus any intermediate
    waypoints of the segment.
    """
    step = traj[:child_index]
    if prefix is None or prefix.shape[0] == 0:
        return np.array(step, copy=True)
    return np.concatenate([prefix, step], axis=0)


def build_inpaint_prior(prefixes: list, states: np.ndarray, H: int, k: int):
    """Prior + mask for PREFIX-INPAINTED expansion (Diffusion-Forcing-inspired).

    Glue-mode expansion samples continuations conditioned only on the leaf state
    and concatenates them onto the search prefix; the seam is exactly where the
    critic's off-manifold error lives. Inpaint mode instead clamps the WHOLE
    prefix (plus the node state) into the denoiser at every diffusion step —
    the same conditioning-by-replacement mechanism the DV planner already uses
    for row 0 — so the free rows are generated jointly consistent with the path
    and the sampled window IS the composed [s0, s0+H) window, seam-free.

    prefixes: per-node (d_i, D) arrays (or None at the root), states: (B, D)
    node states, k: samples per node. Returns (prior (B*k, H, D),
    mask (B*k, H, D), d_lens (B,)) — block i*k:(i+1)*k belongs to node i, rows
    [0:d_i] carry the prefix, row d_i the node state, and the mask marks those
    d_i+1 rows as fixed. Raises if a prefix leaves no free rows to plan.
    """
    states = np.asarray(states, dtype=np.float32)
    B, D = states.shape
    prior = np.zeros((B * k, H, D), dtype=np.float32)
    mask = np.zeros((B * k, H, D), dtype=np.float32)
    d_lens = np.zeros(B, dtype=np.int64)
    for i in range(B):
        p = prefixes[i]
        d = 0 if p is None else int(np.asarray(p).shape[0])
        if d + 1 >= H:
            raise ValueError(
                f"inpaint prefix length {d} leaves no free rows in horizon {H}; "
                f"keep budget * child_index <= H - 1")
        blk = slice(i * k, (i + 1) * k)
        if d:
            prior[blk, :d] = np.asarray(p, dtype=np.float32)
        prior[blk, d] = states[i]
        mask[blk, : d + 1] = 1.0
        d_lens[i] = d
    return prior, mask, d_lens
