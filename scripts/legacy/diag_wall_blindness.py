"""scripts/diag_wall_blindness.py — does V(s,g) "see" walls it was never shown?

The robot never observes the maze map: the planner/policy/value are trained only on
state trajectories, so V(s,g) learns distance-to-goal from relabelled travel TIME in
the data. Where the data covers a region it can learn the wall-respecting (geodesic)
distance; where it must extrapolate ACROSS a wall it has no signal and tends to fall
back on something Euclidean — judging a goal "close" when a wall makes it far. That is
exactly the error the oracle (true BFS geodesic) does not make, and the candidate
mis-ranking it would cause (a goalward-LOOKING endpoint that is behind a wall).

This probe quantifies and pictures that gap, offline (no rollouts):
  * evaluate V(s,g) on a broad sample of dataset states toward the fixed eval goal;
  * compare to the TRUE BFS geodesic (oracle, dev-only) on the SAME value scale;
  * report whether V(s,g) tracks the geodesic or the Euclidean distance, and how
    over-optimistic it is for high-detour (wall-between) states.

Outputs:
  * stats (corr with geodesic vs Euclidean; over-optimism for high- vs low-detour)
  * {out_fig}/wallblind_error_map.png   — per-cell mean (V_pred - geodesic-value);
    red = V(s,g) thinks it is closer than it is (wall-blind over-optimism), walls drawn
  * {out_fig}/wallblind_implied_vs_geodesic.png — implied vs true distance, coloured by
    Euclidean detour (points below the diagonal with low Euclidean = wall-blind)

Oracle (walls/geodesics) is dev-only (Rule 1); all numbers DIAGNOSTIC-ONLY.

Run:
    python scripts/diag_wall_blindness.py --env antmaze-large-diverse-v2 \
        --sg-ckpt state_value_sg_ckpt_best.pt --out-fig results/instr/figs
"""
import argparse
import json
import os
import sys

sys.path.insert(0, ".")

import numpy as np
import torch

from mcts.maze_oracle import AntMazeOracle, calibrate_steps_per_cell
from mcts.mcts_loop import load_models
from mcts.relabel import build_relabel_inputs
from mcts.specs import get_goal, make_dataset, normalize_goal_xy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="antmaze-large-diverse-v2")
    p.add_argument("--sg-ckpt", default="state_value_sg_ckpt_best.pt")
    p.add_argument("--n-states", type=int, default=4000,
                   help="dataset states sampled for broad maze coverage")
    p.add_argument("--pess", action="store_true",
                   help="use ensemble-min (the deployed pessimistic value) instead of mean")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None, help="JSON stats path")
    p.add_argument("--out-fig", default="results/instr/figs")
    args = p.parse_args()

    models = load_models(args.env, device=args.device, ckpt_dir=args.ckpt,
                         sg_ckpt=args.sg_ckpt)
    if models.get("value_sg") is None:
        sys.exit("no V(s,g) ensemble loaded -- pass --sg-ckpt <trained checkpoint>")
    dev = models["device"]
    ens, normalizer = models["value_sg"], models["normalizer"]
    torch.manual_seed(args.seed)

    env, ds = make_dataset(args.env)
    oracle = AntMazeOracle(env)
    env.reset()
    goal_xy = np.asarray(get_goal(env), dtype=np.float64)
    goal_norm = normalize_goal_xy(normalizer, goal_xy)
    goal_grid = np.array(oracle.dist_grid_from(goal_xy))

    seq_obs, ends, term_only, scale = build_relabel_inputs(ds)
    if ens.meta.get("D") and int(ens.meta["D"]) != int(scale.D):
        sys.exit(f"value-scale mismatch D {ens.meta['D']} vs {scale.D}")
    path_xys = [normalizer.unnormalize(seq_obs[i, :ends[i] + 1])[:, :2]
                for i in range(len(ends))]
    spc = calibrate_steps_per_cell(oracle, path_xys, ends)
    print(f"goal={goal_xy.round(2)}  steps/cell={spc:.1f}  D={scale.D}  "
          f"value={'pess(min)' if args.pess else 'mean'}")

    # ── sample dataset states, score V(s,g) toward the eval goal ─────────────────
    rng = np.random.default_rng(args.seed)
    picks = []
    for _ in range(args.n_states):
        p = int(rng.integers(0, len(ends)))
        picks.append(seq_obs[p, int(rng.integers(0, ends[p] + 1))])
    states = np.stack(picks).astype(np.float32)                  # (N, D) normed
    xy = normalizer.unnormalize(states)[:, :2]                   # (N, 2) world
    s = torch.as_tensor(states, device=dev)
    g = torch.as_tensor(goal_norm, dtype=torch.float32, device=dev).expand(len(states), 2)
    with torch.no_grad():
        x = torch.cat([s, g], dim=-1)
        v = (ens.pessimistic(x, mode="min").squeeze(-1) if args.pess
             else ens(x).mean(dim=-1)).cpu().numpy()             # (N,) predicted value

    # ── true geodesic / Euclidean on the same value scale ───────────────────────
    cells = [oracle.cell(pt) for pt in xy]
    geo_cells = np.array([goal_grid[r][c] for r, c in cells], dtype=float)  # inf if behind a wall
    in_wall = np.array([oracle.wall[r][c] for r, c in cells])
    euclid_cells = np.linalg.norm(xy - goal_xy, axis=1) / oracle.scaling
    valid = np.isfinite(geo_cells) & ~in_wall & (euclid_cells > 1e-6)

    val_geo = scale.val_array(geo_cells * spc)                   # value the geodesic implies
    implied_cells = ((1.0 - v) * scale.D / 2.0) / spc            # value -> implied cells
    error = v - val_geo                                          # >0 == V(s,g) over-optimistic
    detour = np.where(valid, geo_cells / np.maximum(euclid_cells, 1.0), np.nan)

    m = valid
    corr_geo = float(np.corrcoef(v[m], geo_cells[m])[0, 1])
    corr_euc = float(np.corrcoef(v[m], euclid_cells[m])[0, 1])
    hi = detour[m] >= np.nanpercentile(detour[m], 80)            # wall-between states
    overopt_hi = float(error[m][hi].mean())
    overopt_lo = float(error[m][~hi].mean())
    gap = overopt_hi - overopt_lo
    verdict = ("WALL-BLIND -- over-optimistic exactly where walls intervene"
               if gap > 0.05 else "not clearly wall-blind (detour gap small)")
    print("\n" + "=" * 70)
    print("WALL-BLINDNESS — V(s,g) vs true geodesic (dataset states, eval goal)")
    # PRIMARY signal: the detour-stratified over-optimism gap. It cancels any global
    # steps/cell calibration bias and isolates the extrapolation error (the value is
    # wall-blind only where it must cross an un-traversed wall), so it is the verdict.
    print("  PRIMARY (bias-robust) — over-optimism (value - geodesic-value):")
    print(f"     high-detour (wall-between) states: {overopt_hi:+.3f}")
    print(f"     low-detour  (open) states        : {overopt_lo:+.3f}")
    print(f"     gap (hi - lo)                    : {gap:+.3f}   => {verdict}")
    # SUPPORTING colour only: geodesic and Euclidean are highly collinear, so the
    # corr comparison can flip on noise — do not read it as the verdict.
    print(f"  supporting — corr(value, geodesic)={corr_geo:+.3f} vs "
          f"corr(value, euclidean)={corr_euc:+.3f}  (collinear; colour, not verdict)")
    # LOCALIZED: the detour gap is a global average that cancels over-optimistic (red)
    # cells against over-conservative (blue) ones. Report the per-cell over-optimism
    # directly — the spatially-concentrated wall-blind spots the heatmap reveals.
    csum, ccnt = {}, {}
    for (r, c), e, ok in zip(cells, error, valid):
        if ok:
            csum[(r, c)] = csum.get((r, c), 0.0) + e
            ccnt[(r, c)] = ccnt.get((r, c), 0) + 1
    cellerr = np.array([csum[k] / ccnt[k] for k in csum]) if csum else np.array([0.0])
    frac_red = float((cellerr > 0.10).mean())
    max_red = float(cellerr.max())
    print(f"  LOCALIZED — {100*frac_red:.0f}% of visited cells are over-optimistic "
          f"(mean err > 0.10), worst cell +{max_red:.2f}.")
    print(f"     => the global gap masks real wall-blind pockets; see the error heatmap.")
    print("=" * 70)

    stats = dict(env=args.env, sg_ckpt=args.sg_ckpt, value="pess" if args.pess else "mean",
                 n_valid=int(m.sum()), steps_per_cell=float(spc),
                 overopt_gap=gap, overopt_high_detour=overopt_hi,
                 overopt_low_detour=overopt_lo, verdict=verdict,
                 frac_overoptimistic_cells=frac_red, max_cell_overopt=max_red,
                 corr_geodesic=corr_geo, corr_euclidean=corr_euc,
                 DIAGNOSTIC_ONLY=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"wrote {args.out}")

    _plots(oracle, goal_xy, cells, error, geo_cells, implied_cells, euclid_cells,
           valid, args.out_fig)


def _plots(oracle, goal_xy, cells, error, geo_cells, implied_cells, euclid_cells,
           valid, out_fig):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (matplotlib unavailable: {e!r} — stats only, no figures)")
        return
    os.makedirs(out_fig, exist_ok=True)
    nr, nc = oracle.n_rows, oracle.n_cols
    wall = np.array([[1.0 if w else 0.0 for w in row] for row in oracle.wall])

    # ── error heatmap (mean over-optimism per cell) ─────────────────────────────
    err_sum = np.zeros((nr, nc)); cnt = np.zeros((nr, nc))
    for (r, c), e, ok in zip(cells, error, valid):
        if ok:
            err_sum[r, c] += e; cnt[r, c] += 1
    mean_err = np.where(cnt > 0, err_sum / np.maximum(cnt, 1), np.nan)
    lim = float(np.nanmax(np.abs(mean_err))) if np.isfinite(mean_err).any() else 1.0
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(wall, cmap="Greys", origin="upper", interpolation="nearest", alpha=0.25)
    im = ax.imshow(np.ma.masked_invalid(mean_err), cmap="RdBu_r", origin="upper",
                   vmin=-lim, vmax=lim, interpolation="nearest")
    gr, gc = oracle.cell(goal_xy)
    ax.scatter([gc], [gr], c="lime", marker="*", s=240, edgecolor="k", zorder=5)
    # outline walls
    for r in range(nr):
        for c in range(nc):
            if wall[r, c]:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                           edgecolor="k", lw=0.6))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="mean (V(s,g) - geodesic value)  [red = over-optimistic]")
    ax.set_title("Wall-blindness: where V(s,g) thinks it is closer than it is\n"
                 "(red cells behind walls = goal looks near but is geodesically far)")
    fig.tight_layout()
    p1 = os.path.join(out_fig, "wallblind_error_map.png")
    fig.savefig(p1, dpi=120); plt.close(fig)

    # ── implied vs true geodesic, coloured by Euclidean detour ──────────────────
    m = valid
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sc = ax.scatter(geo_cells[m], implied_cells[m], c=euclid_cells[m], cmap="viridis",
                    s=8, alpha=0.4, linewidths=0)
    hi = max(np.nanmax(geo_cells[m]), np.nanmax(implied_cells[m]))
    ax.plot([0, hi], [0, hi], "k--", lw=1, label="perfect (implied = true)")
    ax.set_xlabel("true BFS geodesic to goal (cells)")
    ax.set_ylabel("V(s,g)-implied distance (cells)")
    ax.set_title("Below the line + low Euclidean = wall-blind\n"
                 "(value says 'close' while the geodesic says 'far')")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Euclidean to goal (cells)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    p2 = os.path.join(out_fig, "wallblind_implied_vs_geodesic.png")
    fig.savefig(p2, dpi=120); plt.close(fig)
    print(f"wrote {p1}\nwrote {p2}")


if __name__ == "__main__":
    main()
