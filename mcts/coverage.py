"""mcts/coverage.py

Trajectory two-point coverage — the connectivity stratum for D1 (plan v5.1, R5.1).

A query (s, g) is **coverable** if some single dataset trajectory passes within
eps of BOTH s and g (in xy) — i.e. the within-trajectory relabeling of §3a could
have produced a training pair like it. Otherwise it is **stitched**: the critic's
output there is pure extrapolation, the regime D4 is structurally blind to and
the pre-registered trigger for the IQL-u upgrade.

Implementation: xy space is gridded at cell size eps; each path contributes its
visited cells to an inverted index cell -> {path ids}. "Within eps of a visited
point" is covered by scanning the 3x3 cell neighbourhood of the query (any point
within eps of a visited point lies within one cell of it when cells have side eps).
coverable(s, g) = the near-path sets of s and g intersect.

Pure stdlib (dicts/sets of ints) so the stratum logic is unit-tested on the
torch-free local box; building from ~1.5M dataset points takes seconds on the
GPU box and is done once per diagnostic run.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Set, Tuple

Cell = Tuple[int, int]


class TrajectoryCoverage:
    def __init__(self, eps: float) -> None:
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.eps = float(eps)
        self._cell_to_paths: Dict[Cell, Set[int]] = {}
        self.n_paths = 0

    def _cell(self, x: float, y: float) -> Cell:
        return (int(math.floor(x / self.eps)), int(math.floor(y / self.eps)))

    def add_path(self, path_id: int, xys: Iterable) -> None:
        """Register one trajectory's visited xy points (any iterable of (x, y))."""
        index = self._cell_to_paths
        for xy in xys:
            c = self._cell(float(xy[0]), float(xy[1]))
            s = index.get(c)
            if s is None:
                index[c] = {path_id}
            else:
                s.add(path_id)
        self.n_paths = max(self.n_paths, path_id + 1)

    def paths_near(self, xy) -> Set[int]:
        """Path ids passing within ~eps of xy (3x3 cell neighbourhood)."""
        cx, cy = self._cell(float(xy[0]), float(xy[1]))
        out: Set[int] = set()
        index = self._cell_to_paths
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                s = index.get((cx + dx, cy + dy))
                if s:
                    out |= s
        return out

    def coverable(self, s_xy, g_xy) -> bool:
        """True iff one single trajectory passes within eps of BOTH points."""
        near_s = self.paths_near(s_xy)
        if not near_s:
            return False
        return not near_s.isdisjoint(self.paths_near(g_xy))

    def stratum(self, s_xy, g_xy) -> str:
        return "coverable" if self.coverable(s_xy, g_xy) else "stitched"

    def stats(self) -> dict:
        return dict(n_paths=self.n_paths, n_cells=len(self._cell_to_paths),
                    eps=self.eps)
