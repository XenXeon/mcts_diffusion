"""scripts/diag_d4_calibration.py

D4 — no-oracle calibration of V(s, g) (plan v5.1 §3).

On relabelled (state, future-goal) pairs from HELD-OUT paths, compare the
critic's predicted step distance d̂ = steps(V) against the exact within-trajectory
step distance, by distance band. Reports MAE and signed bias (negative bias =
critic thinks it is closer than it is — the dangerous direction for the tree).

⚠ KNOWN BLIND SPOT (disclosed, R5.1): these pairs are within-trajectory by
construction — D4 can only validate the regime the §3a relabeling trained on.
The stitched regime is covered exclusively by D1's strata 2–3
(scripts/diag_d1_compass.py). D4 is the no-oracle diagnostic that travels;
D1-stitched is the one that protects deployment.

Run on the GPU box:
    python scripts/diag_d4_calibration.py --env antmaze-large-diverse-v2
"""
import argparse
import json
import sys

sys.path.insert(0, ".")

import numpy as np
import torch

from mcts.relabel import build_relabel_inputs, path_val_split, sample_batch
from mcts.specs import ckpt_dir, make_dataset
from mcts.value_net import load_value_ensemble

BANDS = [(0, 50), (50, 100), (100, 200), (200, 400), (400, 700), (700, 10**9)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="antmaze-large-diverse-v2")
    p.add_argument("--sg-ckpt", type=str, default="state_value_sg_ckpt_latest.pt")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--n-pairs", type=int, default=50000)
    p.add_argument("--geo-mean", type=float, default=200.0)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0,
                   help="must match training so the SAME paths are held out")
    p.add_argument("--full-data", choices=["auto", "yes", "no"], default="auto",
                   help="match the held-out split to the critic's training data; "
                        "'auto' reads full_data from the sg checkpoint (audit D1-2)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # Critic first — its meta dictates the data regime and the held-out split, so
    # 'held-out' is genuinely the paths the critic did NOT train on (audit D1-2).
    net = load_value_ensemble(f"{ckpt_dir(args.env, args.ckpt)}/{args.sg_ckpt}",
                              device=device)
    if args.full_data == "auto":
        full_data = bool(net.meta.get("full_data", False))
    else:
        full_data = (args.full_data == "yes")
    # The held-out split is (num_paths, val_frac, seed); auto-detect matches
    # num_paths via the regime, and val_frac/seed come from the critic's meta so a
    # non-default training knob can't silently re-open the leakage (R-C). geo_mean
    # too, so the held-out pair distribution matches what the critic trained on.
    eff_seed = int(net.meta.get("seed", args.seed))
    eff_val_frac = float(net.meta.get("val_frac", args.val_frac))
    eff_geo_mean = float(net.meta.get("geo_mean", args.geo_mean))
    print(f"critic data regime: {'FULL-DATA' if full_data else 'terminus-only'} "
          f"(held-out split matched — D1-2)")
    if (eff_seed, eff_val_frac, eff_geo_mean) != (args.seed, args.val_frac, args.geo_mean):
        print(f"  split/sampling matched to checkpoint: seed={eff_seed} "
              f"val_frac={eff_val_frac} geo_mean={eff_geo_mean} "
              f"(CLI {args.seed}/{args.val_frac}/{args.geo_mean} overridden)")

    env, ds = make_dataset(args.env, learn_policy=full_data)
    seq_obs, ends, term_only, scale = build_relabel_inputs(ds)
    ckpt_D = net.meta.get("D")
    if ckpt_D is not None and int(ckpt_D) != int(scale.D):
        sys.exit(f"value-scale mismatch: critic D={ckpt_D} vs dataset D={scale.D}")

    # The SAME path-level split as the trainer (shared helper, critic's seed + N).
    val_paths, _ = path_val_split(seq_obs.shape[0], eff_val_frac, eff_seed)
    s, g, t = sample_batch(seq_obs, ends, scale, args.n_pairs, eff_geo_mean,
                           np.random.default_rng(eff_seed + 1), paths=val_paths)
    actual_steps = scale.steps(t.squeeze(-1))            # shared affine (C2)
    with torch.no_grad():
        x = torch.tensor(np.concatenate([s, g], axis=-1), device=device)
        v_min = net.pessimistic(x, mode="min").squeeze(-1).cpu().numpy()
        v_mean = net(x).mean(dim=-1).cpu().numpy()
    pred_min = scale.steps(v_min)
    pred_mean = scale.steps(v_mean)

    print(f"D4 calibration — {args.env}  (n={args.n_pairs} held-out relabelled "
          f"pairs, D={scale.D})")
    hdr = (f"{'band (steps)':>14} {'n':>6} {'MAE min':>8} {'bias min':>9} "
           f"{'MAE mean':>9} {'bias mean':>10}")
    print(hdr); print("-" * len(hdr))
    results = {}
    for lo, hi in BANDS:
        m = (actual_steps >= lo) & (actual_steps < hi)
        if m.sum() < 20:
            continue
        row = dict(n=int(m.sum()),
                   mae_min=float(np.abs(pred_min[m] - actual_steps[m]).mean()),
                   bias_min=float((pred_min[m] - actual_steps[m]).mean()),
                   mae_mean=float(np.abs(pred_mean[m] - actual_steps[m]).mean()),
                   bias_mean=float((pred_mean[m] - actual_steps[m]).mean()))
        results[f"{lo}-{hi}"] = row
        label = f"{lo}-{hi if hi < 10**9 else 'max'}"
        print(f"{label:>14} {row['n']:>6} {row['mae_min']:>8.0f} "
              f"{row['bias_min']:>+9.0f} {row['mae_mean']:>9.0f} "
              f"{row['bias_mean']:>+10.0f}")

    print("\nNOTE (R5.1): within-trajectory pairs only — this diagnostic is "
          "structurally blind to the stitched regime; see diag_d1_compass.py "
          "strata 2-3 for the deployment-protecting check.")
    out = args.out or f"results/d4_calibration_{args.env}.json"
    with open(out, "w") as f:
        json.dump(dict(env=args.env, full_data=full_data, sg_ckpt=args.sg_ckpt,
                       D=scale.D, results=results, args=vars(args)), f, indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
