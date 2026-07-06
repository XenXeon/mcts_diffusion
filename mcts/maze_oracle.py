"""mcts/maze_oracle.py

Dev-only geodesic oracle for D4RL antmaze (plan v5.1 §1 Rule 1).

⚠ ORACLE DISCIPLINE: this module may be used for diagnostics (D1 strata, D2
enrichment), development gates, and evaluation-only measurement. It must NEVER
be imported by any component whose numbers reach the results table, nor touch
the training data of any learned component (classifier negatives included).

Adapted from the validated maze2d BFS oracle in scripts/phase6_stage0_oracle.py.
d4rl antmaze (locomotion) exposes the maze as `_maze_map` (list of lists; 1=wall,
0=free, 'r'=reset, 'g'=goal) with `_maze_size_scaling` (4.0 for large) and world
offsets `_init_torso_x/_init_torso_y` (world xy = col*scaling - init_x,
row*scaling - init_y). Attribute names vary slightly across d4rl versions, so
lookup is defensive and `ascii_map()` exists to eyeball the transform — run it
once on the GPU box before trusting any geodesic number.

Geodesics are in CELL units; convert to dense steps with `steps_per_cell`
calibrated from data (calibrate_steps_per_cell), never assumed.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

Cell = Tuple[int, int]


def _get_attr(obj, names, what):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    raise AttributeError(f"could not locate {what} on {type(obj).__name__} "
                         f"(tried {names}); inspect the env and extend the list")


class AntMazeOracle:
    def __init__(self, env) -> None:
        u = env.unwrapped
        maze = _get_attr(u, ("_maze_map", "maze_map", "_np_maze_map"), "maze map")
        self.scaling = float(_get_attr(
            u, ("_maze_size_scaling", "maze_size_scaling"), "maze scaling"))
        self.init_x = float(_get_attr(u, ("_init_torso_x",), "init torso x"))
        self.init_y = float(_get_attr(u, ("_init_torso_y",), "init torso y"))
        self.wall: List[List[bool]] = [
            [str(c) == "1" or c == 1 for c in row] for row in maze]
        self.n_rows, self.n_cols = len(self.wall), len(self.wall[0])
        self._bfs_cache: Dict[Cell, List[List[float]]] = {}

    # ── coordinate transform ───────────────────────────────────────────────────
    def cell(self, xy) -> Cell:
        col = int(round((float(xy[0]) + self.init_x) / self.scaling))
        row = int(round((float(xy[1]) + self.init_y) / self.scaling))
        return (min(max(row, 0), self.n_rows - 1),
                min(max(col, 0), self.n_cols - 1))

    def xy(self, cell: Cell):
        r, c = cell
        return (c * self.scaling - self.init_x, r * self.scaling - self.init_y)

    # ── BFS geodesics (cell units) ─────────────────────────────────────────────
    def _bfs(self, src: Cell) -> List[List[float]]:
        if src in self._bfs_cache:
            return self._bfs_cache[src]
        dist = [[math.inf] * self.n_cols for _ in range(self.n_rows)]
        if not self.wall[src[0]][src[1]]:
            dist[src[0]][src[1]] = 0.0
            q = deque([src])
            while q:
                r, c = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < self.n_rows and 0 <= nc < self.n_cols
                            and not self.wall[nr][nc]
                            and dist[nr][nc] == math.inf):
                        dist[nr][nc] = dist[r][c] + 1.0
                        q.append((nr, nc))
        self._bfs_cache[src] = dist
        return dist

    def geodesic_cells(self, a_xy, b_xy) -> float:
        """BFS cell distance between two world points (inf if separated)."""
        return self._bfs(self.cell(a_xy))[self.cell(b_xy)[0]][self.cell(b_xy)[1]]

    def dist_grid_from(self, xy) -> List[List[float]]:
        """Full BFS grid from a point (one call per goal in D1 — cached)."""
        return self._bfs(self.cell(xy))

    def dmax_cells(self, from_xy) -> float:
        grid = self._bfs(self.cell(from_xy))
        return max(d for row in grid for d in row if math.isfinite(d))

    # ── rendering for the mandatory transform check ────────────────────────────
    def ascii_map(self, marks: Optional[Dict[str, Sequence[float]]] = None) -> str:
        """'#'=wall, '.'=free, plus single-char marks at given world xy points."""
        grid = [["#" if w else "." for w in row] for row in self.wall]
        for ch, xy in (marks or {}).items():
            r, c = self.cell(xy)
            grid[r][c] = ch[0]
        return "\n".join(" ".join(row) for row in grid)


class Maze2DOracle:
    """Geodesic oracle for D4RL maze2d (point mass), same interface as AntMazeOracle so the
    diag_oracle_flat / animator tooling is family-agnostic. Reuses the VALIDATED transform from
    scripts/phase6_stage0_oracle.py (BFSValue): wall = (maze_arr == 10) [d4rl WALL], and
    xy->cell via the env's canonical `_xy_to_rowcol` if present, else (row=round(y), col=round(x))
    — the maze2d coordinate frame is already ~cell-indexed. scaling=1, init=0 so the shared
    animator formula col=(x+init_x)/scaling matches. ALWAYS eyeball ascii_map() on the box first
    (the medium/large transforms were not re-validated after umaze). Rule-1 DIAGNOSTIC-ONLY."""

    def __init__(self, env) -> None:
        import numpy as np
        u = env.unwrapped
        arr = np.asarray(_get_attr(u, ("maze_arr", "_maze_arr"), "maze array"))
        if arr.dtype.kind in "iuf":
            wall = (arr == 10)                                   # d4rl WALL marker
        else:
            wall = np.isin(arr, np.array(["#", "x", "1", b"#"], dtype=arr.dtype))
        self.wall: List[List[bool]] = [[bool(w) for w in row] for row in wall]
        self.n_rows, self.n_cols = len(self.wall), len(self.wall[0])
        self._xy_to_rc = getattr(u, "_xy_to_rowcol", None)
        self.scaling, self.init_x, self.init_y = 1.0, 0.0, 0.0   # for the shared animator overlay
        self._bfs_cache: Dict[Cell, List[List[float]]] = {}

    def cell(self, xy) -> Cell:
        if self._xy_to_rc is not None:
            import numpy as np
            rc = self._xy_to_rc(np.asarray(xy))
            r, c = int(round(float(rc[0]))), int(round(float(rc[1])))
        else:
            # maze2d convention: row<-x, col<-y (OPPOSITE of antmaze's row<-y,col<-x).
            # DEFINITIVE (maze2d-large, 8 resets): goal [7,9] -> cell (7,9) = the '12' marker
            # at maze_arr[7][9], and all 8 start positions land on free cells this way, while
            # row<-y,col<-x put 12/16 in walls. umaze/medium are diagonal so don't distinguish.
            r, c = int(round(float(xy[0]))), int(round(float(xy[1])))   # row=x, col=y
        return (min(max(r, 0), self.n_rows - 1), min(max(c, 0), self.n_cols - 1))

    def xy(self, cell: Cell):
        r, c = cell
        return (float(r), float(c))                             # inverse of cell: x=row, y=col

    # BFS geodesics (cell units) — identical to AntMazeOracle, kept standalone for safety.
    def _bfs(self, src: Cell) -> List[List[float]]:
        if src in self._bfs_cache:
            return self._bfs_cache[src]
        dist = [[math.inf] * self.n_cols for _ in range(self.n_rows)]
        if not self.wall[src[0]][src[1]]:
            dist[src[0]][src[1]] = 0.0
            q = deque([src])
            while q:
                r, c = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < self.n_rows and 0 <= nc < self.n_cols
                            and not self.wall[nr][nc] and dist[nr][nc] == math.inf):
                        dist[nr][nc] = dist[r][c] + 1.0
                        q.append((nr, nc))
        self._bfs_cache[src] = dist
        return dist

    def geodesic_cells(self, a_xy, b_xy) -> float:
        return self._bfs(self.cell(a_xy))[self.cell(b_xy)[0]][self.cell(b_xy)[1]]

    def dist_grid_from(self, xy) -> List[List[float]]:
        return self._bfs(self.cell(xy))

    def dmax_cells(self, from_xy) -> float:
        grid = self._bfs(self.cell(from_xy))
        return max(d for row in grid for d in row if math.isfinite(d))

    def ascii_map(self, marks: Optional[Dict[str, Sequence[float]]] = None) -> str:
        grid = [["#" if w else "." for w in row] for row in self.wall]
        for ch, xy in (marks or {}).items():
            r, c = self.cell(xy)
            grid[r][c] = ch[0]
        return "\n".join(" ".join(row) for row in grid)


def make_oracle(env, family: str):
    """Family-dispatched geodesic oracle (Rule-1 DIAGNOSTIC-ONLY)."""
    return Maze2DOracle(env) if family == "maze2d" else AntMazeOracle(env)


def calibrate_steps_per_cell(oracle: AntMazeOracle, path_xys, terminus,
                             max_pairs: int = 2000, min_cells: int = 3,
                             rng=None) -> float:
    """Median dense-steps per BFS cell, measured from real trajectories.

    For random within-trajectory index pairs (t, t') the dense-step gap is
    exactly t'-t; divide by the BFS cell distance between the two positions.
    Geodesics in steps are then cells * steps_per_cell — calibrated, not assumed.
    """
    import random as _random
    rng = rng or _random.Random(0)
    ratios: List[float] = []
    n_paths = len(path_xys)
    while len(ratios) < max_pairs:
        p = rng.randrange(n_paths)
        T = terminus[p]
        if T < 20:
            continue
        t = rng.randrange(0, T - 10)
        tp = rng.randrange(t + 10, T + 1)
        g = oracle.geodesic_cells(path_xys[p][t], path_xys[p][tp])
        if math.isfinite(g) and g >= min_cells:
            ratios.append((tp - t) / g)
    ratios.sort()
    return ratios[len(ratios) // 2]
