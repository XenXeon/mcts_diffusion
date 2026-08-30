"""scripts/diag_d1_compass.py

D1 — compass resolution, connectivity-stratified (plan v5.1 §3, R5.1/R5.3).

Measures Δ*: the smallest geodesic gap at which the critic ranks far-zone state
pairs (which of s_a, s_b is closer to goal g) with ≥80% accuracy — reported in
three strata:
    coverable : both (s, g) queries are within-trajectory-coverable (some single
                dataset trajectory passes within eps of both points) — the regime
                the §3a relabeling actually trained on;
    stitched  : at least one query is NOT coverable — pure extrapolation, the
                regime D4 is structurally blind to. THE pre-registered trigger:
                if this stratum fails, step (b) IQL-u is required;
    corner    : goals sampled around the eval corner (the deployed distribution),
                with its own coverable/stitched split.

Gates: the ONLY hard gate is the cell-D (L4) unlock — stitched AND corner
Δ* ≤ 100 dense steps. For L1 cells D1 is informative, never blocking (Phase-1 ran
L1 to 83.3% with a critic that fails far-zone 25-step resolution; depth converts
near-goal sharpness into far-zone discrimination).

Oracle discipline: BFS geodesics and the coverage index are dev-only (Rule 1).

Run on the GPU box (after training the V(s,g) ensemble):
    python scripts/diag_d1_compass.py --env antmaze-large-diverse-v2
    python scripts/diag_d1_compass.py --env antmaze-large-diverse-v2 --diagnose
"""
import argparse
import json
import math
import sys

sys.path.insert(0, ".")

import numpy as np
import torch

from mcts.coverage import TrajectoryCoverage
from mcts.maze_oracle import AntMazeOracle, calibrate_steps_per_cell
from mcts.relabel import build_relabel_inputs
from mcts.specs import ckpt_dir, get_goal, make_dataset, normalize_goal_xy
from mcts.value_net import load_state_value, load_value_ensemble

ACC_GATE, MIN_N = 0.80, 50
# The single far-zone resolution threshold (dense steps). BOTH the L4-unlock gate
# and the IQL-u trigger read the stitched stratum against THIS number — they are
# two views of one decision (§3b: "step (b) triggers when D1's stitched stratum
# fails its gate"), so they must never use different thresholds (B1).
DELTA_STAR_GATE = 100


def proprio_bins(raw_states: np.ndarray) -> np.ndarray:
    """Bin antmaze states by (torso z, speed, |quat w|) so pairs are
    posture-matched and the critic cannot rank on proprioception shortcuts.
    Antmaze obs layout: [x, y, z, quat(4), joints(8), qvel(14)]; vx, vy = 15:17."""
    z = raw_states[:, 2]
    speed = np.hypot(raw_states[:, 15], raw_states[:, 16])
    upright = np.abs(raw_states[:, 3])
    def tercile(v):
        lo, hi = np.quantile(v, [1 / 3, 2 / 3])
        return (v > lo).astype(int) + (v > hi).astype(int)
    return tercile(z) * 6 + tercile(speed) * 2 + (upright > np.median(upright))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="antmaze-large-diverse-v2")
    p.add_argument("--sg-ckpt", type=str, default="state_value_sg_ckpt_latest.pt")
    p.add_argument("--vs-ckpt", type=str, default="state_value_ckpt_latest.pt")
    p.add_argument("--ckpt", type=str, default=None, help="override ckpt dir base")
    p.add_argument("--bands", type=int, nargs="+",
                   default=[25, 50, 100, 150, 250])
    p.add_argument("--far-zone", type=float, default=400.0,
                   help="both states at least this many dense steps from goal")
    p.add_argument("--pairs-per-cell", type=int, default=500)
    p.add_argument("--n-goals", type=int, default=150)
    p.add_argument("--n-corner-goals", type=int, default=60)
    p.add_argument("--state-stride", type=int, default=10,
                   help="subsample every k-th dataset state as pair candidates")
    p.add_argument("--eps", type=float, default=0.5,
                   help="coverage radius (= goal radius)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--diagnose", action="store_true",
                   help="print maze + transform marks and exit")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--full-data", choices=["auto", "yes", "no"], default="auto",
                   help="match coverage/candidates to the critic's training data; "
                        "'auto' reads full_data from the sg checkpoint (audit D1-1)")
    args = p.parse_args()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    # ── critics first: the sg checkpoint's meta dictates the data regime ─────────
    base = ckpt_dir(args.env, args.ckpt)
    sg = load_value_ensemble(f"{base}/{args.sg_ckpt}", device=device)
    if args.full_data == "auto":
        full_data = bool(sg.meta.get("full_data", False))
    else:
        full_data = (args.full_data == "yes")
    print(f"critic data regime: {'FULL-DATA' if full_data else 'terminus-only'} "
          f"(coverage + candidates built to match — D1-1)")
    try:
        vs = load_state_value(f"{base}/{args.vs_ckpt}", device=device)
    except FileNotFoundError:
        vs = None
        print("(no V(s) checkpoint found — baseline column skipped)")

    # ── data, oracle, coverage (matched to the critic) ───────────────────────────
    env, ds = make_dataset(args.env, learn_policy=full_data)
    oracle = AntMazeOracle(env)
    env.reset()
    corner = np.asarray(get_goal(env), dtype=np.float64)
    if args.diagnose:
        print(oracle.ascii_map(marks={"S": (0, 0), "G": corner}))
        return

    normalizer = ds.get_normalizer()
    # Shared derivation: 'stitched' = not coverable by ANY training trajectory of
    # THIS critic (terminus + timeout in full-data), so the stratum stays "pure
    # extrapolation for this critic" and the L4 gate / IQL-u trigger read honestly.
    seq_obs, ends, term_only, scale = build_relabel_inputs(ds)
    ckpt_D = sg.meta.get("D")
    if ckpt_D is not None and int(ckpt_D) != int(scale.D):
        sys.exit(f"value-scale mismatch: critic trained with D={ckpt_D} but this "
                 f"dataset re-derives D={scale.D} — diagnostics would mis-rank. "
                 f"Check the data regime / path_end_indices.")
    cov = TrajectoryCoverage(eps=args.eps)
    path_xys, cand_norm, cand_raw = [], [], []
    for p_idx in range(seq_obs.shape[0]):
        T = ends[p_idx]
        raw = normalizer.unnormalize(seq_obs[p_idx, :T + 1])
        path_xys.append(raw[:, :2])
        cov.add_path(p_idx, raw[:, :2])
        sel = np.arange(0, T + 1, args.state_stride)
        cand_norm.append(seq_obs[p_idx, sel])
        cand_raw.append(raw[sel])
    cand_norm = np.concatenate(cand_norm)            # (N, obs_dim) normalised
    cand_raw = np.concatenate(cand_raw)              # (N, obs_dim) raw
    bins = proprio_bins(cand_raw)
    spc = calibrate_steps_per_cell(oracle, path_xys, ends)
    print(f"candidates={len(cand_norm)}  paths={len(path_xys)}  "
          f"steps/cell={spc:.1f}  coverage={cov.stats()}")

    # ── pair construction ───────────────────────────────────────────────────────
    # buckets[(stratum, band)] -> list of (idx_a, idx_b, goal_xy); a is CLOSER.
    # Candidate cells are precomputed ONCE; per-goal geodesics are then a single
    # numpy gather into that goal's BFS grid.
    cand_cells = np.array([oracle.cell(xy) for xy in cand_raw[:, :2]])  # (N, 2)
    buckets: dict = {}
    all_keys = ([("coverable", b) for b in args.bands]
                + [("stitched", b) for b in args.bands]
                + [(f"corner{suf}", b) for b in args.bands
                   for suf in ("", "_coverable", "_stitched")])

    def full():
        return all(len(buckets.get(k, [])) >= args.pairs_per_cell
                   for k in all_keys)

    goal_sets = ([("data", tuple(cand_raw[i, :2]))
                  for i in rng.integers(0, len(cand_raw), args.n_goals)]
                 + [("corner", tuple(corner + rng.uniform(-0.75, 0.75, 2)))
                    for _ in range(args.n_corner_goals)])
    for gkind, g_xy in goal_sets:
        if full():
            break
        grid = np.array(oracle.dist_grid_from(g_xy))
        geo = grid[cand_cells[:, 0], cand_cells[:, 1]] * spc
        far = np.isfinite(geo) & (geo >= args.far_zone)
        idx = np.flatnonzero(far)
        if len(idx) < 2:
            continue
        for band in args.bands:
            lo, hi = 0.8 * band, 1.2 * band
            take = rng.choice(idx, size=min(len(idx), 50), replace=False)
            for i in take:
                # candidate partners: same proprio bin, gap within the band
                gap = np.abs(geo - geo[i])
                ok = far & (bins == bins[i]) & (gap >= lo) & (gap <= hi)
                ok[i] = False
                js = np.flatnonzero(ok)
                if len(js) == 0:
                    continue
                j = int(rng.choice(js))
                a, b = (i, j) if geo[i] < geo[j] else (j, i)
                cov_a = cov.coverable(cand_raw[a, :2], g_xy)
                cov_b = cov.coverable(cand_raw[b, :2], g_xy)
                stratum = "coverable" if (cov_a and cov_b) else "stitched"
                keys = ([stratum] if gkind == "data"
                        else ["corner", f"corner_{stratum}"])
                for key in keys:
                    bucket = buckets.setdefault((key, band), [])
                    if len(bucket) < args.pairs_per_cell:
                        bucket.append((a, b, g_xy))

    # ── scoring ────────────────────────────────────────────────────────────────
    def rank_acc(pairs, scorer):
        ia = np.array([p[0] for p in pairs]); ib = np.array([p[1] for p in pairs])
        # shared goal normaliser (C1) — identical to training and the sampler
        g = normalize_goal_xy(normalizer, np.array([p[2] for p in pairs]))
        va = scorer(cand_norm[ia], g)
        vb = scorer(cand_norm[ib], g)
        return float(np.mean(va > vb))             # a is closer ⇒ should score higher

    @torch.no_grad()
    def score_sg(s, g):
        x = torch.tensor(np.concatenate([s, g], 1), dtype=torch.float32,
                         device=device)
        return sg.pessimistic(x, mode="min").squeeze(-1).cpu().numpy()

    @torch.no_grad()
    def score_vs(s, g):
        x = torch.tensor(s, dtype=torch.float32, device=device)
        return vs(x).squeeze(-1).cpu().numpy()

    results = {}
    hdr = f"{'stratum':>16} {'band':>5} {'n':>5} {'acc V(s,g)min':>13} {'acc V(s)':>9}"
    print("\n" + hdr); print("-" * len(hdr))
    strata = sorted({k for k, _ in buckets})
    for stratum in strata:
        for band in args.bands:
            pairs = buckets.get((stratum, band), [])
            if not pairs:
                continue
            acc = rank_acc(pairs, score_sg)
            acc_b = rank_acc(pairs, score_vs) if vs is not None else float("nan")
            results[f"{stratum}|{band}"] = dict(n=len(pairs), acc_sg=acc,
                                                acc_vs=acc_b)
            print(f"{stratum:>16} {band:>5} {len(pairs):>5} {acc:>13.3f} "
                  f"{acc_b:>9.3f}")

    def delta_star(stratum):
        for band in sorted(args.bands):
            r = results.get(f"{stratum}|{band}")
            if r and r["n"] >= MIN_N and r["acc_sg"] >= ACC_GATE:
                return band
        return None

    print("\nΔ* (smallest band with acc ≥ 0.8, n ≥ 50):")
    ds_out = {}
    for stratum in strata:
        d = delta_star(stratum)
        ds_out[stratum] = d
        print(f"  {stratum:>16}: {d if d is not None else 'NOT RESOLVED'}")

    st, co = ds_out.get("stitched"), ds_out.get("corner")
    G = DELTA_STAR_GATE

    def passes(d):                # resolved AND within the far-zone gate
        return d is not None and d <= G

    unlocked = passes(st) and passes(co)
    print(f"\nGATE — cell D (L4) unlock (stitched & corner Δ* ≤ {G}): "
          f"{'PASS — L4 interaction cell unlocked' if unlocked else 'FAIL — locked'}")
    # IQL-u trigger reads the SAME threshold (B1): the stitched stratum failing
    # its gate — unresolved OR resolved but > G — is exactly the far-zone failure
    # §3b says must escalate. Previously this fired only on fully-unresolved,
    # silently under-triggering a stitched Δ* in (G, max_band].
    if not passes(st):
        why = "unresolved" if st is None else f"Δ*={st} > {G}"
        print(f"TRIGGER — stitched stratum fails its gate ({why}): step (b) IQL-u "
              f"is required (pre-registered, plan v5.1 §3b).")
    print("Reminder: D1 is informative-NEVER-blocking for the L1 cells (R5.3).")

    out = args.out or f"results/d1_compass_{args.env}.json"
    with open(out, "w") as f:
        json.dump(dict(env=args.env, full_data=full_data, sg_ckpt=args.sg_ckpt,
                       bands=args.bands, far_zone=args.far_zone,
                       steps_per_cell=spc, results=results, delta_star=ds_out,
                       l4_unlocked=bool(unlocked), args=vars(args)), f, indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
