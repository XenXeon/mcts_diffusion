"""mcts/stitch.py

Torch-free stitched-window construction for SEARCH-COMPATIBLE critic training
(Lever A of the value-improvement plan).

Why
---
The tree scores COMPOSED windows — a search-chosen prefix from one plan spliced
onto a continuation from another (mcts/window.py) — but DV's trajectory critic
was trained only on contiguous dataset windows. The measured winner's curse
(max-backup promotes overrated stitched plans; top-3 backup recovered +4.5,
p<1e-4) is the critic's off-manifold error on exactly those inputs. This module
manufactures stitched training windows WITH EXACT LABELS so the critic can be
fine-tuned on the distribution the tree actually queries.

Label semantics (must replicate DV_D4RLMaze2DSeqDataset EXACTLY)
----------------------------------------------------------------
The dataset's target is seq_val[p, t] = discounted return of the (reward-tuned,
padded) reward stream from dense step t to the end of the padded path:

    seq_val = copy(seq_rew)                                # length L_pad rows
    for i in reversed(range(max_path_length - 1)):
        seq_val[:, i] = seq_rew[:, i] + discount * seq_val[:, i+1]
    seq_val = minmax(seq_val over the FULL array) [* 2 - 1]

(d4rl_maze2d_dataset.py:165-178; seq_rew is ALREADY reward-tuned — iql: raw-1 —
and padded rows carry the tuned pad reward.) `recompute_raw_values` replicates
that recursion bit-for-bit (same float32, same update range, same min/max over
the full array including the never-recursed tail), and the fine-tune script
asserts the replica matches ds.seq_val before training.

The stitched label then needs NO reward re-summation, via the segment identity

    sum_{t=0}^{n-1} g^t r_A[sa+t]  =  V_A[sa] - g^n * V_A[sa+n]

so a window that follows path A for n = j*stride dense steps and then path B
from a matching state sb has raw label

    V_A[sa] - g^n * V_A[sa+n] + g^n * V_B[sb]

normalised with the SAME dataset min/max. Validity requires sa+n and sb to be
inside the recursion range (<= max_path_length-1) and on REAL (unpadded) steps.

Scope: maze2d family ONLY. The antmaze dataset class builds its paths/padding
differently (noreaching_penalty, no strided tail, path_lengths bookkeeping);
extending this module there requires re-deriving the label recursion against
d4rl_antmaze_dataset.py first — do not duck-type it.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def recompute_raw_values(seq_rew: np.ndarray, discount: float,
                         max_path_length: int) -> Tuple[np.ndarray, float, float]:
    """Replicate the dataset's seq_val recursion on the stored (tuned) rewards.

    seq_rew: (P, L_pad, 1) float32 — ds.seq_rew AS STORED (reward_tune applied).
    Returns (raw_val (P, L_pad, 1) float32, vmin, vmax) where vmin/vmax are taken
    over the FULL array — including the tail rows beyond max_path_length-1 that
    the dataset never recurses — exactly like d4rl_maze2d_dataset.py:170-174.
    """
    raw = np.copy(seq_rew).astype(np.float32)
    for i in reversed(range(max_path_length - 1)):
        raw[:, i] = seq_rew[:, i] + discount * raw[:, i + 1]
    return raw, float(raw.min()), float(raw.max())


def normalize_val(raw, vmin: float, vmax: float, center_mapping: bool = True):
    """The dataset's min-max mapping: [vmin, vmax] -> [0, 1] (-> [-1, 1])."""
    out = (raw - vmin) / (vmax - vmin)
    return out * 2.0 - 1.0 if center_mapping else out


class JunctionIndex:
    """Grid-hash over normalised states for near-match junction lookup.

    Indexes (path, t) entries with t on a REAL step and inside the recursion
    range: t <= min(path_length-1, max_path_length-1). Matching is L-inf <= eps
    over `match_dims` (default: all dims — for maze2d that is [x, y, vx, vy],
    so junctions match in position AND velocity).
    """

    def __init__(self, seq_obs: np.ndarray, path_lengths: Sequence[int],
                 max_path_length: int, eps: float,
                 match_dims: Optional[Sequence[int]] = None,
                 paths: Optional[Sequence[int]] = None,
                 t_subsample: int = 1) -> None:
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")
        self.eps = float(eps)
        self.dims = np.asarray(match_dims if match_dims is not None
                               else range(seq_obs.shape[-1]), dtype=np.int64)
        use_paths = np.asarray(paths if paths is not None
                               else range(seq_obs.shape[0]), dtype=np.int64)

        ps, ts = [], []
        for p in use_paths:
            hi = min(int(path_lengths[p]) - 1, max_path_length - 1)
            if hi < 0:
                continue
            t = np.arange(0, hi + 1, t_subsample, dtype=np.int64)
            ps.append(np.full(t.shape, p, dtype=np.int64))
            ts.append(t)
        if not ps:
            raise ValueError("no indexable states (check paths/path_lengths)")
        self.entry_p = np.concatenate(ps)
        self.entry_t = np.concatenate(ts)
        pts = seq_obs[self.entry_p, self.entry_t][:, self.dims]
        self._pts = pts.astype(np.float32)

        cells = np.floor(pts / self.eps).astype(np.int64)          # (N, d)
        uniq, inverse = np.unique(cells, axis=0, return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        self._order = order
        bounds = np.searchsorted(inverse[order], np.arange(uniq.shape[0] + 1))
        self._cell_slice: Dict[Tuple[int, ...], Tuple[int, int]] = {
            tuple(uniq[u]): (int(bounds[u]), int(bounds[u + 1]))
            for u in range(uniq.shape[0])}
        d = len(self.dims)
        self._offsets = np.stack(np.meshgrid(*([np.array([-1, 0, 1])] * d),
                                             indexing="ij"), -1).reshape(-1, d)

    def __len__(self) -> int:
        return int(self.entry_p.shape[0])

    def query(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """All indexed (paths, ts) with L-inf distance <= eps from `state`."""
        q = np.asarray(state, dtype=np.float32)[self.dims]
        base = np.floor(q / self.eps).astype(np.int64)
        rows: List[np.ndarray] = []
        for off in self._offsets:
            sl = self._cell_slice.get(tuple(base + off))
            if sl is not None:
                rows.append(self._order[sl[0]:sl[1]])
        if not rows:
            return (np.empty(0, dtype=np.int64),) * 2
        idx = np.concatenate(rows)
        keep = np.abs(self._pts[idx] - q).max(axis=-1) <= self.eps
        idx = idx[keep]
        return self.entry_p[idx], self.entry_t[idx]


class StitchSpace:
    """Composed windows + exact labels over one DV maze2d dataset.

    All arrays come from the dataset object (already reward-tuned / normalised):
        seq_obs (P, L_pad, D) normalised states, seq_rew (P, L_pad, 1) tuned
        rewards, path_lengths[p] real (unpadded) steps per stored row.
    """

    def __init__(self, seq_obs: np.ndarray, seq_rew: np.ndarray,
                 path_lengths: Sequence[int], horizon: int, stride: int,
                 max_path_length: int, discount: float = 1.0,
                 center_mapping: bool = True) -> None:
        if horizon < 2:
            raise ValueError(f"horizon must be >= 2, got {horizon}")
        self.seq_obs = np.asarray(seq_obs)
        self.path_lengths = np.asarray(path_lengths, dtype=np.int64)
        self.H, self.stride = int(horizon), int(stride)
        self.L = int(max_path_length)
        self.discount = float(discount)
        self.center_mapping = center_mapping
        self.raw_val, self.vmin, self.vmax = recompute_raw_values(
            np.asarray(seq_rew), self.discount, self.L)

    # ── label machinery ─────────────────────────────────────────────────────────
    def normalize(self, raw):
        return normalize_val(raw, self.vmin, self.vmax, self.center_mapping)

    def consistency_max_err(self, ds_seq_val: np.ndarray) -> float:
        """max |replica - dataset| over the full seq_val array (sanity gate)."""
        return float(np.abs(self.normalize(self.raw_val)
                            - np.asarray(ds_seq_val)).max())

    def stitched_label(self, a: int, sa: int, j: int, b: int, sb: int) -> float:
        """Normalised label of: j waypoints along A from sa, then B from sb.

        n = j*stride dense steps along A, then B's return-to-go at sb. Requires
        sa+n and sb inside the recursion range and on real steps (the sampler
        enforces this; this method only guards the recursion range).
        """
        n = j * self.stride
        if not (1 <= j <= self.H - 1):
            raise ValueError(f"j must be in [1, {self.H - 1}], got {j}")
        if sa + n > self.L - 1 or sb > self.L - 1:
            raise ValueError("junction outside the value-recursion range")
        g = self.discount ** n
        raw = (self.raw_val[a, sa, 0] - g * self.raw_val[a, sa + n, 0]
               + g * self.raw_val[b, sb, 0])
        return float(self.normalize(raw))

    def compose_obs(self, a: int, sa: int, j: int, b: int, sb: int) -> np.ndarray:
        """(H, D) window: j strided waypoints of A from sa, then H-j of B from sb."""
        s = self.stride
        rows_a = self.seq_obs[a, sa:sa + j * s:s]
        rows_b = self.seq_obs[b, sb:sb + (self.H - j) * s:s]
        return np.concatenate([rows_a, rows_b], axis=0)

    # ── batch sampler ───────────────────────────────────────────────────────────
    def sample_stitched(self, rng: np.random.Generator, batch: int,
                        index: JunctionIndex, paths: Optional[Sequence[int]] = None,
                        j_min: int = 1, j_max: Optional[int] = None,
                        min_gap: Optional[int] = None, max_tries: int = 200,
                        ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Draw `batch` stitched windows + labels.

        A-side draws come from `paths` (default: all); B-side candidates come
        from `index` (build it over the same path subset to keep train/val
        splits leak-free). Near-identity stitches — same path with
        |sb - junction| < min_gap (default stride) — are excluded: those are
        just the original continuation, not an augmentation.
        Returns (obs (B, H, D) float32, labels (B, 1) float32, stats).
        """
        a_paths = np.asarray(paths if paths is not None
                             else range(self.seq_obs.shape[0]), dtype=np.int64)
        j_max = j_max if j_max is not None else self.H - 1
        min_gap = min_gap if min_gap is not None else self.stride
        obs = np.empty((batch, self.H, self.seq_obs.shape[-1]), dtype=np.float32)
        lab = np.empty((batch, 1), dtype=np.float32)
        jdist, filled, tries = [], 0, 0
        while filled < batch:
            tries += 1
            if tries > max_tries * batch:
                raise RuntimeError(
                    f"could not fill batch ({filled}/{batch} after {tries} tries)"
                    f" — eps too small or index too sparse")
            a = int(a_paths[rng.integers(len(a_paths))])
            j = int(rng.integers(j_min, j_max + 1))
            sa_hi = min(int(self.path_lengths[a]) - 1, self.L - 1) - j * self.stride
            if sa_hi < 0:
                continue
            sa = int(rng.integers(0, sa_hi + 1))
            junction = self.seq_obs[a, sa + j * self.stride]
            cand_p, cand_t = index.query(junction)
            if cand_p.size == 0:
                continue
            keep = ~((cand_p == a)
                     & (np.abs(cand_t - (sa + j * self.stride)) < min_gap))
            cand_p, cand_t = cand_p[keep], cand_t[keep]
            if cand_p.size == 0:
                continue
            pick = int(rng.integers(cand_p.size))
            b, sb = int(cand_p[pick]), int(cand_t[pick])
            obs[filled] = self.compose_obs(a, sa, j, b, sb)
            lab[filled, 0] = self.stitched_label(a, sa, j, b, sb)
            jdist.append(float(np.abs(self.seq_obs[b, sb] - junction).max()))
            filled += 1
        stats = dict(mean_junction_linf=float(np.mean(jdist)),
                     accept_rate=batch / max(tries, 1))
        return obs, lab, stats

    def sample_original(self, rng: np.random.Generator, batch: int,
                        indices: Sequence[Tuple[int, int, int]],
                        ds_seq_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Plain dataset windows (the non-stitched half of a fine-tune batch).

        `indices` are ds.indices rows (path, start, end); targets are read from
        ds.seq_val so the original half trains on byte-identical labels to the
        base pipeline.
        """
        picks = rng.integers(len(indices), size=batch)
        obs = np.empty((batch, self.H, self.seq_obs.shape[-1]), dtype=np.float32)
        lab = np.empty((batch, 1), dtype=np.float32)
        for i, k in enumerate(picks):
            p, start, end = indices[int(k)]
            obs[i] = self.seq_obs[p, start:end:self.stride]
            lab[i, 0] = ds_seq_val[p, start, 0]
        return obs, lab
