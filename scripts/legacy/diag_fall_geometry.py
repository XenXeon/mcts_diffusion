"""scripts/diag_fall_geometry.py — do topples happen NEAR WALLS or after SHARP TURNS?

Two physical hypotheses for the residual ~20% (the topples that selection can't fix):
  (H-wall)  the ant clips a wall and tips  -> falls cluster in LOW wall-clearance cells
  (H-turn)  a sudden heading change tips it -> falls follow a SHARP commanded turn

Both are testable from the failure logs we already have on disk (no GPU). This reads the
per-failed-episode npz written by diag_oracle_flat --log / run_instrumentation:
  e{i}_xy (T,2 executed torso) , e{i}_chosen_fw (T,2 commanded steering target)
and the maze geometry in the *_index.json. For each failed episode it:
  * finds the STALL ONSET (longest trailing run of near-zero speed) and splits
    TOPPLE/STALL (it stopped and stayed stopped) from CREEP (kept moving to the horizon),
  * measures wall clearance (world units, nearest wall-cell) along the path,
  * measures the commanded turn angle = angle(prev executed velocity, chosen_fw - xy).
Then it compares onset vs the episode's own moving baseline, pooled per tag.

Read it as: if onset clearance << moving-baseline clearance, H-wall holds; if pre-onset
turn >> baseline turn, H-turn holds. If neither separates, the topple is intrinsic to the
low-level policy (the world-model / sim-lookahead direction), not a selectable geometry.

DIAGNOSTIC-ONLY (consumes privileged-geodesic logs). Pure numpy + stdlib.

    python scripts/diag_fall_geometry.py --tags flatlog_k50gnt0 flatlog_k50gnt70 \
        instr_mcss_critic --seed 0
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np


def _wall_centers(maze):
    """(N,2) array of wall-cell centres in fractional (col,row) grid coords."""
    wall = maze["wall"]
    pts = [(c + 0.5, r + 0.5)
           for r, row in enumerate(wall) for c, w in enumerate(row) if w]
    return np.asarray(pts, dtype=float)


def _colrow(xy, maze):
    col = (xy[..., 0] + maze["init_x"]) / maze["scaling"]
    row = (xy[..., 1] + maze["init_y"]) / maze["scaling"]
    return np.stack([col, row], axis=-1)


def _clearance(xy, maze, wc):
    """World-unit distance from each path point to the nearest wall-cell centre."""
    cr = _colrow(xy, maze)                                   # (T,2) in cell units
    # (T,1,2) - (1,N,2) -> (T,N); min over walls; *scaling -> world units
    d = np.sqrt(((cr[:, None, :] - wc[None, :, :]) ** 2).sum(-1)).min(1)
    return d * maze["scaling"]


def _turn_deg(a, b):
    """Angle (deg) between 2D vectors a,b; nan if either is ~zero length."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return math.nan
    c = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return math.degrees(math.acos(c))


def _stall_onset(speed, thresh, min_len):
    """Index where the trailing low-speed run begins, or None if it never stalls."""
    T = len(speed)
    k = 0
    while k < T and speed[T - 1 - k] < thresh:
        k += 1
    if k >= min_len and k < T:          # stopped and stayed stopped, but did move earlier
        return T - k
    return None


def analyse_tag(in_dir, tag, seed, stall_speed, min_len, pre):
    idxs = glob.glob(os.path.join(in_dir, f"{tag}_s{seed}_index.json"))
    if not idxs:
        print(f"  ({tag}: no index)"); return None
    with open(idxs[0]) as fh:
        idx = json.load(fh)
    npz = np.load(os.path.join(in_dir, idx["npz"]), allow_pickle=False)
    maze = idx["maze"]
    wc = _wall_centers(maze)
    failed = [s for s in idx["scenarios"]
              if not s["success"] and f"e{s['env_idx']}_xy" in npz]

    n_topple = n_creep = 0
    onset_clear, base_clear = [], []          # wall clearance: at-onset vs moving baseline
    pre_turn, base_turn = [], []              # commanded turn: pre-onset vs moving baseline
    for s in failed:
        i = s["env_idx"]
        xy = np.asarray(npz[f"e{i}_xy"], float)
        fw = np.asarray(npz[f"e{i}_chosen_fw"], float) if f"e{i}_chosen_fw" in npz else None
        T = len(xy)
        if T < min_len + pre + 2:
            continue
        v = np.diff(xy, axis=0)                          # (T-1,2) executed step vectors
        speed = np.linalg.norm(v, axis=1)
        clear = _clearance(xy, maze, wc)                 # (T,)
        onset = _stall_onset(speed, stall_speed, min_len)
        # commanded turn at step t: angle(prev executed velocity, fw[t]-xy[t])
        turn = np.full(T, np.nan)
        if fw is not None:
            for t in range(1, T):
                if speed[t - 1] >= stall_speed:
                    turn[t] = _turn_deg(v[t - 1], fw[t] - xy[t])
        moving = speed >= stall_speed                    # mask over steps 0..T-2
        if onset is None:                                # never sustainedly stopped
            n_creep += 1
            continue
        n_topple += 1
        onset_clear.append(float(clear[onset]))
        mv = moving[:onset]
        if mv.any():
            base_clear.append(float(np.median(clear[:onset][mv])))
        w = turn[max(1, onset - pre):onset]
        w = w[np.isfinite(w)]
        if w.size:
            pre_turn.append(float(np.nanmax(w)))         # the sharpest steer entering the stall
        tb = turn[1:onset][np.isfinite(turn[1:onset])]
        if tb.size:
            base_turn.append(float(np.median(tb)))
    return dict(tag=tag, n_failed=len(failed), n_topple=n_topple, n_creep=n_creep,
                onset_clear=onset_clear, base_clear=base_clear,
                pre_turn=pre_turn, base_turn=base_turn)


def _med(x):
    return float(np.median(x)) if len(x) else math.nan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", default="results/instr")
    p.add_argument("--tags", nargs="+",
                   default=["instr_mcss_critic", "flatlog_k50gnt0",
                            "flatlog_k50gnt70", "flatlog_k50stbD2"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stall-speed", type=float, default=0.02,
                   help="world units/step below which the ant counts as not moving")
    p.add_argument("--min-len", type=int, default=25,
                   help="trailing low-speed steps required to call it a stall (vs brief pause)")
    p.add_argument("--pre", type=int, default=6,
                   help="steps before onset to scan for the sharpest commanded turn")
    args = p.parse_args()

    rows = [analyse_tag(args.in_dir, t, args.seed, args.stall_speed, args.min_len, args.pre)
            for t in args.tags]
    rows = [r for r in rows if r]

    print("\n  STALL/TOPPLE split (a 'topple' stops and stays stopped; 'creep' keeps moving):")
    print(f"  {'tag':22s} {'failed':>6} {'topple':>6} {'creep':>6}")
    for r in rows:
        print(f"  {r['tag']:22s} {r['n_failed']:>6} {r['n_topple']:>6} {r['n_creep']:>6}")

    print("\n  H-wall — wall clearance (world units) at the stall onset vs the ant's own")
    print("  moving baseline.  onset << baseline  =>  topples cluster near walls.")
    print(f"  {'tag':22s} {'onset_clear':>12} {'base_clear':>11} {'ratio':>7}")
    for r in rows:
        oc, bc = _med(r["onset_clear"]), _med(r["base_clear"])
        ratio = oc / bc if bc and math.isfinite(bc) and bc > 0 else math.nan
        print(f"  {r['tag']:22s} {oc:>12.3f} {bc:>11.3f} {ratio:>7.2f}")

    print("\n  H-turn — sharpest commanded turn (deg) entering the stall vs moving-baseline")
    print("  median turn.  pre >> baseline  =>  topples follow a sharp steer.")
    print(f"  {'tag':22s} {'pre_turn':>9} {'base_turn':>10} {'delta':>7}")
    for r in rows:
        pt, bt = _med(r["pre_turn"]), _med(r["base_turn"])
        d = pt - bt if math.isfinite(pt) and math.isfinite(bt) else math.nan
        print(f"  {r['tag']:22s} {pt:>9.1f} {bt:>10.1f} {d:>7.1f}")
    print()


if __name__ == "__main__":
    main()
