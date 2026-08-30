"""scripts/finetune_critic_stitched.py

Lever A — SEARCH-COMPATIBLE critic fine-tuning (maze2d only).

The measured winner's curse (max-backup over stitched composites loses -5.7 vs
MCSS even with a superset root; top-3 backup recovers +4.5, p<1e-4) is the DV
trajectory critic's off-manifold error on stitch-point windows it never saw in
training. This script fine-tunes the critic on a mix of ORIGINAL dataset
windows (byte-identical targets to the base pipeline) and STITCHED windows with
EXACT labels (mcts/stitch.py: segment-return identity on the dataset's own
value recursion).

Safety rails:
  * hard family gate: maze2d only (the antmaze dataset builds paths/padding
    differently — extending needs its own label derivation);
  * self-consistency gate before training: the label replica must match
    ds.seq_val to <= --consistency-tol everywhere, or the script aborts;
  * path-level train/val split (shared RNG convention with train_state_value):
    val stitches join val paths only, so eval windows are leak-free;
  * "before" metrics are printed from the frozen base critic so the fine-tune's
    effect on BOTH stitched and original windows is visible (catastrophic
    forgetting on originals would show up immediately).

Run (GPU box):
    python scripts/finetune_critic_stitched.py --env maze2d-large-v1

Output (co-located with the planner/critic ckpts):
    critic_ckpt_stitched.pt        final step
    critic_ckpt_stitched_best.pt   best val-stitched MSE   <- deploy this one
    critic_stitched_train_log.json
Deploy in the tree:  scripts/run_mcts_compare.py ... --critic-step stitched_best
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn.functional as F

from cleandiffuser.utils import DVHorizonCritic
from mcts.relabel import path_val_split
from mcts.specs import SPECS, env_family, make_dataset
from mcts.stitch import JunctionIndex, StitchSpace
from pipelines.utils import set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, required=True)
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--stitch-frac", type=float, default=0.5,
                   help="fraction of each batch that is stitched windows")
    p.add_argument("--eps", type=float, default=0.05,
                   help="junction match tolerance, L-inf in NORMALIZED units "
                        "over all dims (maze2d: x, y, vx, vy)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base-critic-step", type=int, default=1000000)
    p.add_argument("--out-name", type=str, default="stitched")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--eval-interval", type=int, default=5000)
    p.add_argument("--eval-batch", type=int, default=2048)
    p.add_argument("--t-subsample", type=int, default=1,
                   help="index every n-th dense step (memory/speed knob)")
    p.add_argument("--consistency-tol", type=float, default=1e-3)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    if env_family(args.env) != "maze2d":
        sys.exit("finetune_critic_stitched supports the maze2d family only — the "
                 "antmaze dataset class builds paths/padding differently and needs "
                 "its own label derivation (see mcts/stitch.py docstring).")

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    spec = SPECS["maze2d"]
    env, ds = make_dataset(args.env)
    H, stride, L = ds.horizon, ds.stride, ds.max_path_length
    obs_dim = ds.o_dim
    path_lengths = [e - s + 1 for s, e in ds.paths]
    print(f"[{args.env}] paths={len(path_lengths)} H={H} stride={stride} "
          f"max_path_length={L} obs_dim={obs_dim}")

    # ── label replica + HARD consistency gate ─────────────────────────────────
    space = StitchSpace(np.asarray(ds.seq_obs), np.asarray(ds.seq_rew),
                        path_lengths, H, stride, L,
                        discount=1.0, center_mapping=True)
    ds_seq_val = np.asarray(ds.seq_val)
    err = space.consistency_max_err(ds_seq_val)
    print(f"label replica vs ds.seq_val: max abs err = {err:.2e} "
          f"(tol {args.consistency_tol})")
    if err > args.consistency_tol:
        sys.exit("ABORT: value replica does not match the dataset — the stitched "
                 "labels would be on a different scale. Check TARGET_CFG drift.")

    # ── path-level split; per-split junction indices (no cross-split stitches) ─
    val_paths, tr_paths = path_val_split(len(path_lengths), args.val_frac, args.seed)
    idx_tr = JunctionIndex(space.seq_obs, path_lengths, L, eps=args.eps,
                           paths=tr_paths, t_subsample=args.t_subsample)
    idx_va = JunctionIndex(space.seq_obs, path_lengths, L, eps=args.eps,
                           paths=val_paths, t_subsample=args.t_subsample)
    print(f"junction index: train {len(idx_tr):,} states, val {len(idx_va):,} "
          f"(eps={args.eps}, t_subsample={args.t_subsample})")
    tr_paths_set, va_paths_set = set(tr_paths), set(val_paths)
    ind_tr = [i for i in ds.indices if i[0] in tr_paths_set]
    ind_va = [i for i in ds.indices if i[0] in va_paths_set]

    # ── fixed eval sets (val paths only) ──────────────────────────────────────
    rng_eval = np.random.default_rng(args.seed + 1)
    ev_st_obs, ev_st_lab, st_stats = space.sample_stitched(
        rng_eval, args.eval_batch, idx_va, paths=val_paths)
    ev_or_obs, ev_or_lab = space.sample_original(
        rng_eval, args.eval_batch, ind_va, ds_seq_val)
    print(f"val stitched set: mean junction L-inf = "
          f"{st_stats['mean_junction_linf']:.4f}, accept rate = "
          f"{st_stats['accept_rate']:.2f}")
    ev_st = (torch.tensor(ev_st_obs, device=device),
             torch.tensor(ev_st_lab, device=device))
    ev_or = (torch.tensor(ev_or_obs, device=device),
             torch.tensor(ev_or_lab, device=device))

    # ── critic: identical construction to mcts_loop.load_models ──────────────
    ckpt_dir = (args.ckpt or spec["ckpt"]) + f"/{args.env}"
    critic = DVHorizonCritic(obs_dim, emb_dim=128, d_model=256, n_heads=4,
                             depth=2, norm_type="pre").to(device)
    base = torch.load(f"{ckpt_dir}/critic_ckpt_{args.base_critic_step}.pt",
                      map_location=device, weights_only=False)["critic"]
    critic.load_state_dict(base)

    @torch.no_grad()
    def evaluate():
        critic.eval()
        mse_st = float(F.mse_loss(critic(ev_st[0]), ev_st[1]))
        mse_or = float(F.mse_loss(critic(ev_or[0]), ev_or[1]))
        critic.train()
        return mse_st, mse_or

    mse_st0, mse_or0 = evaluate()
    print(f"BEFORE fine-tune: val MSE stitched={mse_st0:.5f} "
          f"original={mse_or0:.5f}  (ratio {mse_st0 / max(mse_or0, 1e-12):.2f}x)")

    optim = torch.optim.Adam(critic.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.steps)
    rng = np.random.default_rng(args.seed + 2)
    n_st = int(round(args.batch * args.stitch_frac))
    n_or = args.batch - n_st
    log, best = [], (float("inf"), -1)
    t0 = time.time()
    critic.train()
    for step in range(1, args.steps + 1):
        parts_o, parts_l = [], []
        if n_st:
            o, l, _ = space.sample_stitched(rng, n_st, idx_tr, paths=tr_paths)
            parts_o.append(o), parts_l.append(l)
        if n_or:
            o, l = space.sample_original(rng, n_or, ind_tr, ds_seq_val)
            parts_o.append(o), parts_l.append(l)
        obs = torch.tensor(np.concatenate(parts_o), device=device)
        lab = torch.tensor(np.concatenate(parts_l), device=device)
        pred = critic(obs)
        assert pred.shape == lab.shape
        loss = F.mse_loss(pred, lab)
        optim.zero_grad()
        loss.backward()
        optim.step()
        sched.step()

        if step % args.eval_interval == 0 or step == args.steps:
            mse_st, mse_or = evaluate()
            log.append(dict(step=step, mse_stitched=mse_st, mse_original=mse_or,
                            train_loss=float(loss.detach()),
                            lr=sched.get_last_lr()[0]))
            marker = ""
            if mse_st < best[0]:
                best = (mse_st, step)
                torch.save({"critic": critic.state_dict()},
                           f"{ckpt_dir}/critic_ckpt_{args.out_name}_best.pt")
                marker = "  <- best"
            print(f"step {step:>7}  val MSE stitched={mse_st:.5f} "
                  f"original={mse_or:.5f}  ({time.time() - t0:.0f}s){marker}")

    torch.save({"critic": critic.state_dict()},
               f"{ckpt_dir}/critic_ckpt_{args.out_name}.pt")
    payload = dict(env=args.env, args=vars(args), before=dict(
        mse_stitched=mse_st0, mse_original=mse_or0),
        best_step=best[1], best_mse_stitched=best[0], log=log,
        consistency_err=err, n_train_paths=len(tr_paths),
        n_val_paths=len(val_paths))
    with open(f"{ckpt_dir}/critic_{args.out_name}_train_log.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"saved critic_ckpt_{args.out_name}[_best].pt  "
          f"best@{best[1]} stitched MSE {best[0]:.5f} "
          f"(before {mse_st0:.5f}); deploy with --critic-step {args.out_name}_best")


if __name__ == "__main__":
    main()
