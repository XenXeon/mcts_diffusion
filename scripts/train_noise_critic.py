"""scripts/train_noise_critic.py

Train the per-token noise-aware value V(x, k) (mcts/noise_critic.py) used for
classifier guidance (CG) on the FROZEN Causal Diffusion Forcing planner
(mcts/df_model.py), and as a candidate in-tree leaf evaluator. The DF planner
and the DV trajectory critic are NOT retrained — this is a third, independent
model that only reads DF windows and per-token noise levels.

Motivation (notes/methodology_report.md §8.5, design 2 — "CG on frozen DF with
a per-token noise-aware value"): the DV trajectory critic is noise-aware only
at the WHOLE-WINDOW level (one score for a fully-clean trajectory) and cannot
see inside a partially-denoised DF sample, so it cannot steer generation, only
rank finished plans. Classical classifier guidance (Diffuser) is noise-aware
per TRAJECTORY (one t per window); this critic is noise-aware per TOKEN,
matching DF's own asymmetric cleanliness (near-future tokens nearly clean,
far-future tokens noisy under the pyramid schedule, mcts/df_schedule.py). The
label is the SAME target family the DV critic ranks with — the dataset's
normalized discounted return-to-go at the window start (seq_val[path, start],
in [-1, 1] with center_mapping=True, cf. scripts/train_state_value.py) — so
V(x, k) is comparable to the DV critic's V(x, 0) in the k=0 (clean) limit.

Gates before any guidance claim:
  1. --smoke run finishes, loss finite and falling;
  2. eval clean_corr (k=0) should approach the DV critic's own correlation
     with return — a noise-aware value that cannot rank CLEAN windows has no
     hope of guiding noisy ones;
  3. eval sched_corr (the pyramid-schedule + clean-history query distribution
     classifier guidance actually evaluates at during sampling) should be
     comfortably positive before spending compute on a guided closed-loop run
     (scripts/run_mcts_compare.py --df-ckpt ... --cg-ckpt ...).

Run (GPU box; --K MUST match the DF planner checkpoint's K):
    python scripts/train_noise_critic.py --env maze2d-large-v1 --smoke   # ~1 min
    python scripts/train_noise_critic.py --env maze2d-large-v1 --K 20
    python scripts/train_noise_critic.py --env kitchen-mixed-v0 --K 20
"""
import argparse
import json
import math
import sys
import time

sys.path.insert(0, ".")

import numpy as np
import torch

from mcts.df_schedule import sample_training_levels
from mcts.noise_critic import NoiseAwareCritic
from mcts.specs import SPECS, env_family, make_dataset
from pipelines.utils import set_seed


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, 0.0 on a degenerate (zero-variance) side — mirrors
    train_state_value.py's evaluate() so corr numbers read the same way."""
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, required=True)
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--K", type=int, default=20,
                   help="noise levels (0=clean). MUST match the DF planner "
                        "checkpoint's K (mcts/df_model.py) — the critic's "
                        "alpha-bar table is shared with the sampler it guides "
                        "(enforced by Sampler.__init__ in mcts/mcts_loop.py)")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--ema-rate", type=float, default=0.999)
    p.add_argument("--p-sched", type=float, default=0.5,
                   help="mcts/df_schedule.py sample_training_levels: fraction "
                        "of training levels drawn from the pyramid schedule "
                        "(vs i.i.d. uniform per-token)")
    p.add_argument("--p-hist", type=float, default=0.5,
                   help="within pyramid draws, fraction with a clean-history "
                        "prefix (mirrors the tree's clamped search prefix)")
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="fraction of PATHS held out for validation (path-level "
                        "split — windows overlap heavily within a path, so a "
                        "window-level split would leak)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-tag", type=str, default="final",
                   help="saved as noise_critic_ckpt_<tag>.pt")
    p.add_argument("--save-interval", type=int, default=50000)
    p.add_argument("--log-interval", type=int, default=2500)
    p.add_argument("--eval-interval", type=int, default=2500)
    p.add_argument("--smoke", action="store_true",
                   help="200 steps + tiny eval: shapes/finiteness gate")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    fam = env_family(args.env)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    if args.smoke:
        args.steps, args.save_interval, args.log_interval = 200, 10**9, 50
        args.eval_interval = 50
        # a 200-step smoke model must NEVER clobber a real checkpoint (same
        # guard as scripts/train_df_planner.py) — always safe to run on any env.
        if not args.out_tag.endswith("_smoke"):
            args.out_tag += "_smoke"

    env, ds = make_dataset(args.env)
    H, stride, D = ds.horizon, ds.stride, ds.o_dim
    seq_obs = np.asarray(ds.seq_obs)                     # (P, L+pad, D) normalized
    idx = np.asarray([(i[0], i[1]) for i in ds.indices], dtype=np.int64)
    # Labels: the SAME return-to-go the dataset __getitem__ exposes as 'val'
    # (seq_val[path_idx, start]) — the DV critic's ranking target. Not every
    # dataset family carries it (e.g. a raw-trajectory-only variant), so fail
    # loudly with a clear, actionable message rather than train on garbage.
    seq_val = getattr(ds, "seq_val", None)
    if seq_val is None:
        sys.exit(f"[{args.env}] dataset class {type(ds).__name__} exposes no "
                 f"seq_val — the noise-aware critic needs the discounted "
                 f"return-to-go target. Datasets that DO expose it: "
                 f"DV_D4RLMaze2DSeqDataset, DV_D4RLAntmazeSeqDataset, "
                 f"DV_D4RLKitchenSeqDataset (cleandiffuser/dataset/). Check "
                 f"--env resolves to one of the families in mcts/specs.py SPECS.")
    seq_val = np.asarray(seq_val)
    print(f"[{args.env}] {len(idx):,} windows, H={H} stride={stride} D={D}, "
          f"K={args.K}, d_model={args.d_model} depth={args.depth}")

    # ── train/val split BY PATH (not by window): windows overlap heavily within
    # a path (stride-spaced starts of the same trajectory), so a window-level
    # split leaks — a "held-out" window can share almost all its rows with a
    # training window from the same path. ──────────────────────────────────────
    split_rng = np.random.default_rng(args.seed)
    paths = np.unique(idx[:, 0])
    perm = split_rng.permutation(len(paths))
    n_val_paths = math.ceil(args.val_frac * len(paths))
    val_paths = set(paths[perm[:n_val_paths]].tolist())
    is_val = np.isin(idx[:, 0], list(val_paths))
    tr_idx, va_idx = idx[~is_val], idx[is_val]
    print(f"paths: total={len(paths)} val={len(val_paths)}  "
          f"windows: train={len(tr_idx)} val={len(va_idx)}")

    critic = NoiseAwareCritic(D, K=args.K, d_model=args.d_model,
                              n_heads=args.n_heads, depth=args.depth,
                              emb_dim=args.emb_dim, ema_rate=args.ema_rate,
                              device=device)
    # NoiseAwareCritic mirrors DFPlanner's wrapper shape (net + EMA net,
    # cfg dict, ema_update()) — .net is the trainable (non-EMA) module.
    optim = torch.optim.AdamW(critic.net.parameters(), lr=args.lr,
                              weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.steps)
    rng = np.random.default_rng(args.seed + 1)
    ckpt_dir = ((args.ckpt or SPECS[fam]["ckpt"]) + f"/{args.env}")
    out_path = f"{ckpt_dir}/noise_critic_ckpt_{args.out_tag}.pt"
    # the best-ckpt save must respect the smoke guard too — a 200-step smoke
    # run tracking "best" would otherwise clobber the real deployed checkpoint
    best_path = (f"{ckpt_dir}/noise_critic_ckpt_best_smoke.pt"
                 if args.out_tag.endswith("_smoke")
                 else f"{ckpt_dir}/noise_critic_ckpt_best.pt")

    offs = np.arange(H) * stride

    def sample_batch(pool, n):
        sel = pool[rng.integers(len(pool), size=n)]
        # identical gather to dataset __getitem__: seq_obs[p, s:s+(H-1)*stride+1:stride]
        rows = sel[:, 1, None] + offs[None, :]
        x0 = torch.as_tensor(seq_obs[sel[:, 0, None], rows],
                             dtype=torch.float32, device=device)
        # 'val' convention: seq_val[path, start] — the window-START return-to-go
        # (same key the dataset __getitem__ returns), squeezed to (n,).
        val = torch.as_tensor(seq_val[sel[:, 0], sel[:, 1]].reshape(-1),
                              dtype=torch.float32, device=device)
        return x0, val

    # ── fixed validation batch, materialized once (<=2048 windows) ─────────────
    n_val_fixed = min(2048, len(va_idx))
    if n_val_fixed == 0:
        print(f"WARNING: no held-out paths (val_frac={args.val_frac} too small "
              f"for {len(paths)} paths) — eval correlations will be skipped.")
    else:
        val_rng = np.random.default_rng(args.seed + 2)
        val_sel = va_idx[val_rng.choice(len(va_idx), size=n_val_fixed, replace=False)]
        val_rows = val_sel[:, 1, None] + offs[None, :]
        val_x = torch.as_tensor(seq_obs[val_sel[:, 0, None], val_rows],
                                dtype=torch.float32, device=device)
        val_y = seq_val[val_sel[:, 0], val_sel[:, 1]].reshape(-1).astype(np.float32)
        val_k_clean = torch.zeros((n_val_fixed, H), dtype=torch.long, device=device)
        # Fixed seeded draw at p_sched=1.0: the pyramid-schedule + clean-history
        # query distribution classifier guidance is ACTUALLY evaluated on during
        # sampling (see mcts/df_schedule.py sample_training_levels docstring) —
        # this is the guidance-relevant correlation, distinct from the k=0 clean
        # case a plain trajectory critic would also get right.
        val_k_sched = torch.as_tensor(
            sample_training_levels(args.K, H, n_val_fixed,
                                   np.random.default_rng(args.seed + 999),
                                   p_sched=1.0, p_hist=args.p_hist, slope=1),
            dtype=torch.long, device=device)
        # NOISE the val batch to the sched pattern (fixed eps draw): V must be
        # evaluated on actually-noised inputs — a CLEAN window paired with a
        # noisy-k conditioning is a configuration that occurs neither in
        # training (loss noises x0 to k) nor at inference (guidance queries
        # genuinely-noisy x). The eps draw is fixed so evals are comparable
        # across steps and best-ckpt tracking reflects the model, not eval noise.
        g = torch.Generator().manual_seed(args.seed + 1234)
        eps_fixed = torch.randn(val_x.shape, generator=g).to(device)
        val_x_sched = (critic.sqrt_ab[val_k_sched].unsqueeze(-1) * val_x
                       + critic.sqrt_1mab[val_k_sched].unsqueeze(-1) * eps_fixed)

        @torch.no_grad()
        def evaluate():
            pred_clean = critic.value(val_x, val_k_clean, use_ema=True).cpu().numpy()
            pred_sched = critic.value(val_x_sched, val_k_sched, use_ema=True).cpu().numpy()
            return pearson_corr(pred_clean, val_y), pearson_corr(pred_sched, val_y)

    log, eval_log = [], []
    # The V(s) family in this repo OVERFITS well before --steps (see
    # scripts/train_state_value.py: val_corr peaks ~step 6k then declines) —
    # track the best sched_corr checkpoint and deploy THAT for guidance, not
    # the final/latest one.
    best_sched_corr, best_step = -2.0, 0
    t0 = time.time()
    critic.net.train()
    for step in range(1, args.steps + 1):
        x0, val = sample_batch(tr_idx, args.batch)
        k = torch.as_tensor(
            sample_training_levels(args.K, H, args.batch, rng,
                                   args.p_sched, args.p_hist, slope=1),
            dtype=torch.long, device=device)
        loss = critic.loss(x0, val, k)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.net.parameters(), 10.0)
        optim.step()
        sched.step()
        critic.ema_update()
        if step % args.log_interval == 0 or step == args.steps:
            lv = float(loss.detach())
            log.append(dict(step=step, loss=lv))
            print(f"step {step:>7}  loss={lv:.5f}  ({time.time() - t0:.0f}s)")
            if not np.isfinite(lv):
                sys.exit("ABORT: non-finite loss")
        if n_val_fixed > 0 and (step % args.eval_interval == 0 or step == args.steps):
            cc, sc = evaluate()
            eval_log.append(dict(step=step, clean_corr=cc, sched_corr=sc))
            print(f"  eval {step:>7}  clean_corr={cc:+.3f}  sched_corr={sc:+.3f}")
            if sc > best_sched_corr:
                best_sched_corr, best_step = sc, step
                critic.save(best_path, env=args.env,
                            step=step, best_sched_corr=sc, args=vars(args))
        if step % args.save_interval == 0:
            critic.save(f"{ckpt_dir}/noise_critic_ckpt_{step}.pt",
                       env=args.env, step=step, args=vars(args))
    critic.net.eval()
    critic.save(out_path, env=args.env, step=args.steps, args=vars(args))
    print(f"saved -> {out_path}")
    if n_val_fixed > 0:
        print(f"BEST sched_corr={best_sched_corr:.3f} @ step {best_step} -> "
              f"noise_critic_ckpt_best.pt (deploy this for --cg-ckpt, not "
              f"'{args.out_tag}' — same overfit pattern as V(s))")

    with open(f"{ckpt_dir}/noise_critic_train_log_{args.out_tag}.json", "w") as f:
        json.dump(dict(env=args.env, args=vars(args), log=log, eval_log=eval_log,
                       best_step=best_step, best_sched_corr=best_sched_corr),
                 f, indent=2)


if __name__ == "__main__":
    main()
