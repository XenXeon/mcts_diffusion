"""mcts/instrument.py

Tier-0 failure instrumentation for the MCSS baseline, plus the Tier-2 oracle-V
re-rank. Runs on the GPU box (numpy/torch/d4rl); the *analysis* of what it dumps
is the torch-free `mcts/failure_modes.py`.

⚠ ORACLE DISCIPLINE (plan v5.1 Rule 1). This module imports `AntMazeOracle` for
diagnosis only. Everything it produces is DIAGNOSTIC-ONLY and must NEVER reach the
results table: the per-step BFS distances are a measurement aid, and the
`value_source="oracle"` mode (Tier-2 ceiling) uses the geodesic AS the critic,
which is privileged information — it is an upper-bound probe, not a reportable
sampler. The npz/index it writes are tagged DIAGNOSTIC_ONLY accordingly.

What it records (per step, on the rollouts that ultimately FAIL — successes are
dropped to keep storage sane), keyed by scenario index so it pairs against the
existing per-rollout records and against a Tier-2 run on the same seeds:
  * executed torso (x, y) — the ant's actual path
  * body state — torso height, uprightness (up-axis world-z), planar speed
  * BFS cell distance from the executed state to the goal (the progress curve)
  * the full k_mcss candidate pool: each endpoint's (x, y) and BFS distance, the
    DV-critic score, the chosen index, and the value the picker assigned its pick

Determinism note: unlike the production harness (which leaves the diffusion draw
unseeded — see run_episodes), this diagnostic seeds torch so a Tier-1 baseline run
and a Tier-2 oracle run see reproducible rollouts and can be cross-checked
scenario-by-scenario. The env goal draw is already a pure function of --seed.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from mcts.maze_oracle import AntMazeOracle      # pure-stdlib, safe at import
from mcts.specs import env_family, get_goal     # torch-free (heavy deps are lazy)

if TYPE_CHECKING:                                # torch only needed by run_traced
    from mcts.mcts_loop import Sampler

# Keeping torch/cleandiffuser out of the module top (the specs.py pattern) lets the
# analysis side (scripts/analyze_failures.py, plot_failures.py) reuse record_from_npz
# without importing torch — you analyse dumps on a laptop, not the GPU box.


# ── antmaze body-state decoders ──────────────────────────────────────────────────
# d4rl antmaze obs = Ant qpos[15] + qvel[14] = 29 dims. Verified at reset by
# verify_obs_layout (Rule-1-style "eyeball the transform" before trusting it):
#   0,1    torso x, y (world maze plane)
#   2      torso z (height)
#   3..6   torso orientation quaternion (w, x, y, z)   [MuJoCo order]
#   7..14  8 joint angles
#   15..17 torso linear velocity (vx, vy, vz)
#   18..20 torso angular velocity
#   21..28 8 joint velocities

def torso_xy(o: np.ndarray) -> np.ndarray:
    return o[..., 0:2]


def torso_height(o: np.ndarray) -> np.ndarray:
    return o[..., 2]


def uprightness(o: np.ndarray) -> np.ndarray:
    """World-z component of the torso's local up-axis: R22 = 1 - 2(qx^2 + qy^2).

    1.0 == perfectly upright, 0 == on its side, < 0 == flipped. A robust, cheap
    "has the ant toppled" signal that needs only the orientation quaternion.
    """
    qx, qy = o[..., 4], o[..., 5]
    return 1.0 - 2.0 * (qx * qx + qy * qy)


def planar_speed(o: np.ndarray) -> np.ndarray:
    return np.sqrt(o[..., 15] ** 2 + o[..., 16] ** 2)


def verify_obs_layout(obs0: np.ndarray) -> bool:
    """Sanity-check the assumed antmaze obs layout on a reset observation.

    At reset the ant stands upright at the start cell, so torso z is moderate, the
    quaternion is unit-norm, and uprightness ~ 1. If any check fails the body-state
    indices are probably wrong for this d4rl build — warn loudly (don't crash) so
    pose-collapse flags are treated as suspect rather than silently trusted.
    """
    o = np.asarray(obs0, dtype=np.float64).reshape(-1)
    msgs = []
    if o.shape[0] < 17:
        msgs.append(f"obs dim {o.shape[0]} < 17 (not antmaze-shaped)")
    else:
        z = float(o[2])
        qn = float(np.sqrt((o[3:7] ** 2).sum()))
        up = float(1.0 - 2.0 * (o[4] ** 2 + o[5] ** 2))
        if not (0.2 <= z <= 1.5):
            msgs.append(f"torso z={z:.2f} outside [0.2,1.5]")
        if not (0.8 <= qn <= 1.2):
            msgs.append(f"|quat|={qn:.2f} != 1")
        if up < 0.4:
            msgs.append(f"uprightness={up:.2f} < 0.4 at reset")
    if msgs:
        print("  [instrument] WARNING obs-layout check: " + "; ".join(msgs)
              + " -- body-state indices may be wrong for this d4rl build; "
                "treat pose/fall flags as suspect.")
    return not msgs


# ── oracle distance helpers ──────────────────────────────────────────────────────

def _goal_grids(oracle: AntMazeOracle, goals_raw: np.ndarray) -> List[List[List[float]]]:
    """One BFS distance grid per env's goal (cached inside the oracle by source cell)."""
    return [oracle.dist_grid_from(goals_raw[i]) for i in range(len(goals_raw))]


def maze_xy_to_colrow(xy, maze):
    """World xy -> fractional (col, row) on the maze['wall'] grid, for plotting.

    `maze` is the dict stored in each run index (keys scaling/init_x/init_y) — so the
    plotters need neither the env nor the oracle. Mirrors AntMazeOracle.cell's transform
    (col = (x+init_x)/scaling) but keeps the fractional value for smooth overlays.
    """
    xy = np.asarray(xy, dtype=np.float64)
    col = (xy[..., 0] + maze["init_x"]) / maze["scaling"]
    row = (xy[..., 1] + maze["init_y"]) / maze["scaling"]
    return col, row


def _cell_dist(grid: List[List[float]], oracle: AntMazeOracle, xy) -> float:
    r, c = oracle.cell(xy)
    return float(grid[r][c])      # math.inf if in-wall / unreachable from the goal


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


# ── the traced rollout ───────────────────────────────────────────────────────────

def run_traced(sampler: "Sampler", seed: int, n_envs: int,
               max_steps: Optional[int] = None, value_source: str = "critic",
               out_dir: str = "results/instr", tag: Optional[str] = None,
               keep_success_frac: float = 0.0, verbose: bool = True) -> Dict[str, Any]:
    """One paired episode of MCSS (or oracle-ranked MCSS) with full Tier-0 logging.

    value_source:
      * "critic" — the DV trajectory critic argmax (the real MCSS baseline; Tier-0/1).
      * "oracle" — pick the candidate whose ENDPOINT is geodesically closest to the
        goal (Tier-2 ceiling; Rule-1 dev-only, never reportable). All-unreachable
        rows fall back to the critic argmax so the run never stalls.

    Writes (only failed rollouts get heavy traces):
      {out_dir}/{tag}_s{seed}.npz         per-failed-env arrays, keys e{idx}_*
      {out_dir}/{tag}_s{seed}_index.json  per-scenario summary (ALL envs) + run meta
    Returns a summary dict (also the content of the index file).
    """
    import gym
    import torch
    from pipelines.utils import set_seed

    if value_source not in ("critic", "oracle"):
        raise ValueError(f"value_source must be critic|oracle, got {value_source!r}")
    m = sampler.m
    env_name = m["env_name"]
    if env_family(env_name) != "antmaze":
        raise ValueError("instrument.run_traced targets antmaze (the maze with "
                         f"failure headroom); got {env_name}")
    normalizer, env_single = m["normalizer"], m["env_single"]
    obs_dim = m["obs_dim"]
    K = sampler.k_mcss
    max_t = max_steps or m["max_path_length"]
    tag = tag or f"instr_mcss_{value_source}"

    # Reproducible diagnostic study (see module docstring): seed torch too.
    set_seed(seed)
    torch.manual_seed(seed)

    oracle = AntMazeOracle(env_single)
    env = gym.vector.make(env_name, n_envs, asynchronous=False)
    try:
        env.seed(seed)
    except Exception:
        pass
    for i, e in enumerate(getattr(env, "envs", None) or []):
        try:
            e.seed(seed + i)
            e.action_space.seed(seed + i)
        except Exception:
            pass

    obs = env.reset()
    verify_obs_layout(obs[0])
    starts = torso_xy(np.asarray(obs)).astype(np.float64).copy()        # (M,2)
    goals_raw = np.asarray([get_goal(e) for e in env.envs], dtype=np.float64)  # (M,2)
    goal_grids = _goal_grids(oracle, goals_raw)

    # per-env, per-step buffers (appended only while the env is still active)
    buf: List[Dict[str, list]] = [dict(xy=[], dist=[], upright=[], height=[],
                                       speed=[], chosen_value=[], chosen_idx=[],
                                       cand_dist=[], cand_scores=[], cand_xy=[])
                                  for _ in range(n_envs)]
    reach_step = np.full(n_envs, -1, dtype=np.int64)
    success = np.zeros(n_envs, dtype=bool)
    active = np.ones(n_envs, dtype=bool)
    arange = np.arange(n_envs)
    t0 = time.perf_counter()

    for t in range(max_t):
        s_norm = normalizer.normalize(obs).astype(np.float32)           # (M,D)
        prop = sampler.mcss_propose(s_norm)
        scores = prop["scores"]                                         # (M,K)
        endpoints = prop["endpoints"]                                  # (M,K,D)
        first_wps = prop["first_wps"]                                  # (M,K,D)
        # candidate endpoint world xy + BFS distance-to-goal (per env's goal grid)
        ep_xy = normalizer.unnormalize(
            endpoints.reshape(n_envs * K, obs_dim))[:, :2].reshape(n_envs, K, 2)
        cand_dist = np.empty((n_envs, K), dtype=np.float64)
        for i in range(n_envs):
            gi = goal_grids[i]
            for j in range(K):
                cand_dist[i, j] = _cell_dist(gi, oracle, ep_xy[i, j])

        if value_source == "critic":
            chosen = scores.argmax(axis=1)
        else:
            # oracle: nearest endpoint; rows with no reachable candidate -> critic argmax
            safe = np.where(np.isfinite(cand_dist), cand_dist, np.inf)
            chosen = safe.argmin(axis=1)
            allinf = ~np.isfinite(safe).any(axis=1)
            if allinf.any():
                chosen[allinf] = scores[allinf].argmax(axis=1)
        chosen_first_wp = first_wps[arange, chosen]                     # (M,D) normalised
        chosen_value = (scores[arange, chosen] if value_source == "critic"
                        else -cand_dist[arange, chosen])               # higher = better

        # log the decision-time executed state + the pool (only for still-active envs)
        xy_now = torso_xy(np.asarray(obs))
        up_now, h_now, sp_now = uprightness(np.asarray(obs)), \
            torso_height(np.asarray(obs)), planar_speed(np.asarray(obs))
        for i in range(n_envs):
            if not active[i]:
                continue
            b = buf[i]
            b["xy"].append(xy_now[i].astype(np.float32))
            b["dist"].append(_cell_dist(goal_grids[i], oracle, xy_now[i]))
            b["upright"].append(float(up_now[i]))
            b["height"].append(float(h_now[i]))
            b["speed"].append(float(sp_now[i]))
            b["chosen_value"].append(float(chosen_value[i]))
            b["chosen_idx"].append(int(chosen[i]))
            b["cand_dist"].append(cand_dist[i].astype(np.float32))
            b["cand_scores"].append(scores[i].astype(np.float32))
            b["cand_xy"].append(ep_xy[i].astype(np.float32))

        act = sampler.policy_action(s_norm, chosen_first_wp)
        obs, rew, done, info = env.step(act)
        rew = np.asarray(rew, dtype=np.float64)
        hit = active & (rew > 0.0) & (reach_step < 0)
        reach_step[hit] = t
        success[hit] = True
        active &= ~np.asarray(done, dtype=bool)
        if not active.any():
            break
        if verbose and (t + 1) % 100 == 0:
            print(f"  [{tag} s{seed}] t={t+1}/{max_t}  reached={int(success.sum())}/"
                  f"{n_envs}  active={int(active.sum())}  "
                  f"elapsed={time.perf_counter()-t0:.0f}s")

    env.close()
    failed = [i for i in range(n_envs) if not success[i]]
    # Optionally also keep a sample of SUCCESSFUL episodes' traces. Their buffers were
    # recorded up to the reach step, so the data already exists — we just choose to
    # save some, so the analyzer can contrast the critic's mis-rank rate on successes
    # vs failures (the failure-only rate is mildly circular: failures are partly
    # selected by the critic mis-ranking). Sampled with the run seed for reproducibility.
    save_set = list(failed)
    if keep_success_frac > 0.0:
        succ_idx = [i for i in range(n_envs) if success[i]]
        if succ_idx:
            k = min(len(succ_idx), max(1, int(round(keep_success_frac * len(succ_idx)))))
            rs = np.random.default_rng(seed)
            save_set += sorted(int(x) for x in rs.choice(succ_idx, size=k, replace=False))

    # ── persist ──────────────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    arrays: Dict[str, np.ndarray] = {}
    for i in save_set:
        b = buf[i]
        if not b["xy"]:
            continue
        arrays[f"e{i}_xy"] = np.stack(b["xy"])                          # (T,2)
        arrays[f"e{i}_dist"] = np.asarray(b["dist"], dtype=np.float32)  # (T,)
        arrays[f"e{i}_upright"] = np.asarray(b["upright"], dtype=np.float32)
        arrays[f"e{i}_height"] = np.asarray(b["height"], dtype=np.float32)
        arrays[f"e{i}_speed"] = np.asarray(b["speed"], dtype=np.float32)
        arrays[f"e{i}_chosen_value"] = np.asarray(b["chosen_value"], dtype=np.float32)
        arrays[f"e{i}_chosen_idx"] = np.asarray(b["chosen_idx"], dtype=np.int32)
        arrays[f"e{i}_cand_dist"] = np.stack(b["cand_dist"])            # (T,K)
        arrays[f"e{i}_cand_scores"] = np.stack(b["cand_scores"])        # (T,K)
        arrays[f"e{i}_cand_xy"] = np.stack(b["cand_xy"])                # (T,K,2)
    npz_path = os.path.join(out_dir, f"{tag}_s{seed}.npz")
    np.savez_compressed(npz_path, **arrays)

    scenarios = [dict(env_idx=i, seed=int(seed),
                      goal=[float(x) for x in goals_raw[i]],
                      start=[float(x) for x in starts[i]],
                      start_geo_cells=_finite_or_none(
                          _cell_dist(goal_grids[i], oracle, starts[i])),
                      success=bool(success[i]),
                      reach_step=int(reach_step[i]) if reach_step[i] >= 0 else None,
                      n_steps=len(buf[i]["xy"]))
                 for i in range(n_envs)]
    index = dict(
        DIAGNOSTIC_ONLY=True, oracle_used=(value_source == "oracle"),
        rule1_note=("value_source=oracle uses the BFS geodesic as the critic — a "
                    "ceiling probe, never reportable"),
        env=env_name, seed=int(seed), method="mcss", value_source=value_source,
        k_mcss=K, n_envs=n_envs, max_t=max_t,
        reach_pct=float(100.0 * success.mean()), n_failed=len(failed),
        wall_s=round(time.perf_counter() - t0, 1),
        git_commit=_git_commit(), timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        npz=os.path.basename(npz_path),
        # maze geometry so the plotter/analyzer needs no oracle of its own
        maze=dict(wall=[[1 if w else 0 for w in row] for row in oracle.wall],
                  scaling=oracle.scaling, init_x=oracle.init_x, init_y=oracle.init_y,
                  n_rows=oracle.n_rows, n_cols=oracle.n_cols),
        scenarios=scenarios)
    index_path = os.path.join(out_dir, f"{tag}_s{seed}_index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    if verbose:
        print(f"  [{tag} s{seed}] DONE reach={index['reach_pct']:.1f}%  "
              f"failed={len(failed)}/{n_envs}  -> {os.path.basename(npz_path)} "
              f"(+ index)  {index['wall_s']:.0f}s")
    return index


def _finite_or_none(x: float):
    return float(x) if np.isfinite(x) else None


def record_from_npz(npz, i: int, reach_step=None, is_far: bool = False, goal=None):
    """Reconstruct a FailureRecord (mcts.failure_modes) for failed env i from a
    loaded npz, including the candidate pool at the closest-approach (junction) step
    and (given the goal) the min executed WORLD distance for the goal-radius test.

    Shared by scripts/analyze_failures.py and scripts/plot_failures.py so the
    junction logic lives in exactly one place. Returns None if env i has no trace.
    """
    from mcts.failure_modes import FailureRecord, progress_features
    if f"e{i}_dist" not in npz:
        return None
    dist = [float(x) for x in npz[f"e{i}_dist"]]
    upright = [float(x) for x in npz[f"e{i}_upright"]]
    height = [float(x) for x in npz[f"e{i}_height"]]
    f = progress_features(dist)
    jcd = jchosen = None
    if f.argmin_step >= 0 and f"e{i}_cand_dist" in npz:
        s = f.argmin_step
        cand_dist = npz[f"e{i}_cand_dist"][s]
        chosen_j = int(npz[f"e{i}_chosen_idx"][s])
        jcd = [float(x) for x in cand_dist]
        jchosen = float(cand_dist[chosen_j])
    # min executed Euclidean distance to the goal (world units) — the real reward
    # radius test (F1), measured on the dumped path, not the coarse BFS cell grid.
    min_world_dist = None
    if goal is not None and f"e{i}_xy" in npz:
        xy = np.asarray(npz[f"e{i}_xy"], dtype=np.float64)         # (T, 2)
        g = np.asarray(goal, dtype=np.float64).reshape(2)
        if xy.size:
            min_world_dist = float(np.sqrt(((xy - g) ** 2).sum(axis=1)).min())
    return FailureRecord(dist=dist, upright=upright, height=height, success=False,
                         reach_step=reach_step, is_far=is_far,
                         min_world_dist=min_world_dist,
                         junction_cand_dists=jcd, junction_chosen_dist=jchosen)
