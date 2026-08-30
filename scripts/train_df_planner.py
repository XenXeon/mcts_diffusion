"""scripts/train_df_planner.py

Train the Causal Diffusion Forcing planner (mcts/df_model.py) on the SAME
stride-spaced observation windows the DV planner was trained on, for maze2d
and antmaze (kitchen later). The DV critic / inverse-dynamics policy are NOT
retrained — the planner is the only swapped component, so DF arms remain
directly comparable to every DV arm in the study.

Motivation (notes/value_lever_findings.md §5b): prefix-inpainting on the
frozen DV planner is replacement conditioning on an input configuration the
model never saw — measured −16 closed-loop. DF trains on independent
per-token noise levels, making clean-history + noisy-future in-distribution:
tree expansion becomes EXACT conditional generation.

Gates before any tree claim (in order):
  1. --smoke run finishes, loss finite and falling;
  2. after training, the built-in critic check: DF-sampled windows should
     score in the same ballpark as real dataset windows under the DV critic
     (a large gap = weak backbone; tree results would be confounded);
  3. closed-loop DF-MCSS (scripts/run_mcts_compare.py --df-ckpt ...) must
     land within a few points of DV-MCSS before DF-tree vs DF-MCSS means
     anything.

Run (GPU box):
    python scripts/train_df_planner.py --env maze2d-large-v1 --smoke   # ~1 min
    python scripts/train_df_planner.py --env maze2d-large-v1           # hours
    python scripts/train_df_planner.py --env antmaze-large-diverse-v2 --depth 6
"""
import argparse
import json
import sys
import time

sys.path.insert(0, ".")

import numpy as np
import torch

from mcts.df_model import DFPlanner
from mcts.specs import SPECS, env_family, make_dataset
from pipelines.utils import set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, required=True)
    p.add_argument("--steps", type=int, default=400000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--K", type=int, default=20, help="noise levels (0=clean)")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--ema-rate", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-tag", type=str, default="final",
                   help="saved as df_planner_ckpt_<tag>.pt")
    p.add_argument("--save-interval", type=int, default=100000)
    p.add_argument("--log-interval", type=int, default=2500)
    p.add_argument("--smoke", action="store_true",
                   help="200 steps + tiny sample: shapes/finiteness gate")
    p.add_argument("--shortcut", action="store_true",
                   help="train the shortcut-forcing planner (mcts/shortcut_df.py: "
                        "few-step sampling, Dreamer-4 recipe) instead of the "
                        "standard DF planner. Uses weight decay 0.1 (paper: "
                        "crucial). Deploy identically via --df-ckpt <tag>.")
    p.add_argument("--base-units", type=int, default=128,
                   help="shortcut only: smallest-step grid M (dyadic d = 2^j/M)")
    p.add_argument("--sweeps", type=int, default=4,
                   help="shortcut only: default sampling sweeps stored in cfg")
    p.add_argument("--no-critic-check", action="store_true")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    fam = env_family(args.env)
    # kitchen wired 2026-07-07 (DV kitchen checkpoints landed + baseline reproduced at
    # 75.0). DV_D4RLKitchenSeqDataset exposes seq_obs/indices/horizon/stride/o_dim and
    # __getitem__ gathers seq_obs[p, start:start+(H-1)*stride+1:stride] — byte-identical
    # to the sample_batch gather below — so the DF planner trains on the same window
    # distribution as the DV kitchen planner/critic (required for the DV critic to score
    # DF windows). Nothing else is env-specific in this trainer.
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    if args.smoke:
        args.steps, args.save_interval, args.log_interval = 200, 10**9, 50
        # a 200-step smoke model must NEVER clobber a real checkpoint (e.g. the maze2d
        # 'final' behind the confirmed +9.04 result) — write to a disposable tag so
        # `--smoke` is always safe to run on any env.
        if not args.out_tag.endswith("_smoke"):
            args.out_tag += "_smoke"

    env, ds = make_dataset(args.env)
    H, stride, D = ds.horizon, ds.stride, ds.o_dim
    seq_obs = np.asarray(ds.seq_obs)                     # (P, L+pad, D) normalized
    idx = np.asarray([(i[0], i[1]) for i in ds.indices], dtype=np.int64)
    print(f"[{args.env}] {len(idx):,} windows, H={H} stride={stride} D={D}, "
          f"K={args.K}, d_model={args.d_model} depth={args.depth}")

    if args.shortcut:
        from mcts.shortcut_df import ShortcutDFPlanner
        planner = ShortcutDFPlanner(D, base_units=args.base_units,
                                    d_model=args.d_model, n_heads=args.n_heads,
                                    depth=args.depth, ema_rate=args.ema_rate,
                                    default_sweeps=args.sweeps, device=device)
        weight_decay = 0.1     # shortcut paper: crucial for bootstrap stability
    else:
        planner = DFPlanner(D, K=args.K, d_model=args.d_model, n_heads=args.n_heads,
                            depth=args.depth, ema_rate=args.ema_rate, device=device)
        weight_decay = 1e-4
    optim = torch.optim.AdamW(planner.net.parameters(), lr=args.lr,
                              weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.steps)
    rng = np.random.default_rng(args.seed + 1)
    ckpt_dir = ((args.ckpt or SPECS[fam]["ckpt"]) + f"/{args.env}")
    out_path = f"{ckpt_dir}/df_planner_ckpt_{args.out_tag}.pt"

    def sample_batch(n):
        sel = idx[rng.integers(len(idx), size=n)]
        # identical gather to dataset __getitem__: seq_obs[p, s:s+(H-1)*stride+1:stride]
        offs = np.arange(H) * stride
        rows = sel[:, 1, None] + offs[None, :]
        return torch.as_tensor(seq_obs[sel[:, 0, None], rows],
                               dtype=torch.float32, device=device)

    log, t0 = [], time.time()
    planner.net.train()
    for step in range(1, args.steps + 1):
        loss = planner.loss(sample_batch(args.batch))
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(planner.net.parameters(), 10.0)
        optim.step()
        sched.step()
        planner.ema_update()
        if step % args.log_interval == 0 or step == args.steps:
            lv = float(loss.detach())
            log.append(dict(step=step, loss=lv))
            print(f"step {step:>7}  loss={lv:.5f}  ({time.time() - t0:.0f}s)")
            if not np.isfinite(lv):
                sys.exit("ABORT: non-finite loss")
        if step % args.save_interval == 0:
            planner.save(f"{ckpt_dir}/df_planner_ckpt_{step}.pt",
                         env=args.env, step=step, args=vars(args))
    planner.net.eval()
    planner.save(out_path, env=args.env, step=args.steps, args=vars(args))
    print(f"saved -> {out_path}")

    # ── sanity: DF samples vs real windows under the DV critic ──────────────
    n_chk = 16 if args.smoke else 256
    sel = idx[rng.integers(len(idx), size=n_chk)]
    offs = np.arange(H) * stride
    real = torch.as_tensor(
        seq_obs[sel[:, 0, None], sel[:, 1, None] + offs[None, :]],
        dtype=torch.float32, device=device)
    x_hist = torch.zeros_like(real)
    x_hist[:, 0] = real[:, 0]                            # condition on s0 only
    gen = planner.sample(x_hist, torch.ones(n_chk, dtype=torch.long), H)
    hop = lambda w: (w[:, 1:, :2] - w[:, :-1, :2]).norm(dim=-1)
    gh = hop(gen).flatten()
    gen_p99 = gh.kthvalue(max(1, int(0.99 * gh.numel())))[0]
    print(f"xy-hop  real {hop(real).mean():.4f}  gen {hop(gen).mean():.4f} "
          f"(gen p99 {gen_p99:.4f})")
    if not args.no_critic_check:
        try:
            from cleandiffuser.utils import DVHorizonCritic
            critic = DVHorizonCritic(D, emb_dim=128, d_model=256, n_heads=4,
                                     depth=2, norm_type="pre").to(device)
            critic.load_state_dict(torch.load(
                f"{ckpt_dir}/critic_ckpt_1000000.pt", map_location=device,
                weights_only=False)["critic"])
            critic.eval()
            with torch.no_grad():
                sr = critic(real).squeeze(-1)
                sg = critic(gen).squeeze(-1)
            print(f"DV-critic  real {sr.mean():.4f}±{sr.std():.4f}  "
                  f"DF-gen {sg.mean():.4f}±{sg.std():.4f}  "
                  f"(same ballpark = backbone sane; large gap = weak backbone)")
        except Exception as exc:
            print(f"(critic check skipped: {exc!r})")
    with open(f"{ckpt_dir}/df_planner_train_log_{args.out_tag}.json", "w") as f:
        json.dump(dict(env=args.env, args=vars(args), log=log), f, indent=2)


if __name__ == "__main__":
    main()
