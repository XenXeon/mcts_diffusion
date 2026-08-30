"""scripts/diag_inpaint.py

Bug-or-mechanism diagnostic for --expand-mode inpaint (GPU box, ~2 min).

The closed-loop result (maze2d-large r50 MAX: inpaint 182.1 vs glue 198.1,
paired t = -2.84) says prefix-inpainted expansion HURTS. Two explanations:
  (A) BUG — the per-sample fix_mask swap doesn't actually clamp the prefix
      rows, or the child/prefix indexing is off;
  (B) MECHANISM — the clamp works, but the planner (trained with row-0
      clamping only) receives a mixed-noise-level input it never saw:
      clean prefix rows + noisy free rows at a shared t. Off-distribution
      input -> degraded continuations -> misleading critic scores.

This script settles it with numbers, per prefix depth j:
  clamp_err   max |w[:, :j+1] - clamped rows| over samples. ~1e-6 -> the
              mechanism works, (A) is dead. Anything visible -> BUG.
  hop         mean xy step size INSIDE the generated region (rows j+1..),
              vs the same for glue continuations and the dataset's real
              stride-hop reference. Inflated/collapsed hops -> degeneracy.
  seam        xy hop across the junction (row j -> j+1), both modes. Glue's
              seam is the known off-manifold point; inpaint should be smooth
              here IF the planner respects the clamped rows.
  critic      mean +/- std of critic scores, both modes. If inpaint windows
              score systematically HIGHER while being physically worse
              (bigger hops), the tree is being actively misled -> explains
              a below-glue closed loop.

Run:  python scripts/diag_inpaint.py --env maze2d-large-v1
"""
import argparse
import sys

sys.path.insert(0, ".")

import numpy as np
import torch

from mcts.mcts_loop import load_models
from mcts.window import build_inpaint_prior, compose_window
from pipelines.utils import set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="maze2d-large-v1")
    p.add_argument("--n-states", type=int, default=20)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--plan-steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    m = load_models(args.env)
    planner, critic, dev = m["planner"], m["critic"], m["device"]
    H, D = m["H"], m["obs_dim"]
    ds_hop = m.get("wp_disp_sample")
    if ds_hop is not None:
        print(f"dataset stride-hop reference: mean {ds_hop.mean():.4f}, "
              f"p99 {np.percentile(ds_hop, 99):.4f}  (NORMALIZED xy units)")

    def sample_windows(prior, mask=None, k=1):
        n = prior.shape[0]
        if mask is not None:
            base = planner.fix_mask
            planner.fix_mask = mask
        try:
            with torch.no_grad():
                trajs, _ = planner.sample(
                    prior, solver="ddim", n_samples=n, sample_steps=args.plan_steps,
                    use_ema=True, condition_cfg=None, w_cfg=1.0, temperature=1.0)
        finally:
            if mask is not None:
                planner.fix_mask = base
        return trajs

    # root windows from dataset start states — one "search path" source per state
    env_s, ds = m["env_single"], None
    from mcts.specs import make_dataset
    _, ds = make_dataset(args.env)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(ds.indices))[: args.n_states]
    starts = np.stack([np.asarray(ds.seq_obs[ds.indices[i][0], ds.indices[i][1]])
                       for i in idx]).astype(np.float32)
    prior0 = torch.zeros((args.n_states, H, D), device=dev)
    prior0[:, 0, :] = torch.as_tensor(starts, device=dev)
    roots = sample_windows(prior0).cpu().numpy()          # (N, H, D)

    xy_hop = lambda w, a, b: np.linalg.norm(w[..., b, :2] - w[..., a, :2], axis=-1)
    print(f"\n{'j':>3} {'mode':>7} {'clamp_err':>10} {'seam':>8} {'hop':>8} "
          f"{'hop_p99':>8} {'critic':>16}")
    for j in args.depths:
        if j + 1 >= H - 2:
            continue
        prefixes = [roots[i, :j] for i in range(args.n_states)]
        states = roots[:, j]                               # node state = row j

        # ── INPAINT: clamp rows [0:j+1], generate the rest jointly ────────────
        pr, mk, _ = build_inpaint_prior(prefixes, states, H, args.k)
        w_in = sample_windows(torch.as_tensor(pr, device=dev),
                              torch.as_tensor(mk, device=dev)).cpu().numpy()
        w_in = w_in.reshape(args.n_states, args.k, H, D)
        expect = pr.reshape(args.n_states, args.k, H, D)[:, :, : j + 1]
        clamp_err = float(np.abs(w_in[:, :, : j + 1] - expect).max())

        # ── GLUE: continuations from the node state, concatenated ────────────
        prior_g = torch.zeros((args.n_states * args.k, H, D), device=dev)
        prior_g[:, 0, :] = torch.as_tensor(
            np.repeat(states, args.k, axis=0), device=dev)
        cont = sample_windows(prior_g).cpu().numpy().reshape(
            args.n_states, args.k, H, D)
        w_gl = np.stack([compose_window(prefixes[i], cont[i])
                         for i in range(args.n_states)])   # (N, K, H, D)

        with torch.no_grad():
            sc_in = critic(torch.as_tensor(
                w_in.reshape(-1, H, D), device=dev)).squeeze(-1).cpu().numpy()
            sc_gl = critic(torch.as_tensor(
                w_gl.reshape(-1, H, D), device=dev)).squeeze(-1).cpu().numpy()

        for tag, w, sc in (("inpaint", w_in, sc_in), ("glue", w_gl, sc_gl)):
            seam = xy_hop(w, j, j + 1).mean()
            hops = np.stack([xy_hop(w, t, t + 1)
                             for t in range(j + 1, H - 1)])   # generated region
            ce = f"{clamp_err:.2e}" if tag == "inpaint" else "-"
            print(f"{j:>3} {tag:>7} {ce:>10} {seam:>8.4f} {hops.mean():>8.4f} "
                  f"{np.percentile(hops, 99):>8.4f} "
                  f"{sc.mean():>7.4f}±{sc.std():.4f}")
    print("\nREAD: clamp_err ~1e-6 kills the bug theory. Then compare rows: if "
          "inpaint's hop/hop_p99 blow up (degenerate futures) or its critic "
          "scores sit ABOVE glue's while its hops are worse, the planner is "
          "off-distribution under multi-row clamping and the tree is being "
          "misled by inflated scores — mechanism, not bug.")


if __name__ == "__main__":
    main()
