"""scripts/train_dv_classifier.py

Train ONLY the CumRewClassifier for the DV pipeline's original classifier
guidance (guidance_type="cg") on kitchen — the trajectory-level-noise CG the
DV paper compared against MCSS — WITHOUT retraining the planner or policy.

Why this is sound: in `pipelines/veteran_d4rl_kitchen.py` the planner's
diffusion update is IDENTICAL for guidance_type in {MCSS, cg} (unconditional
`planner.update(x)`; only CFG differs), and the policy training is guidance-
agnostic. So the existing MCSS-trained planner/policy checkpoints are exactly
what a cg training run would have produced, and the classifier — which trains
purely off the dataset ((x0 noised by the SDE's own schedule, t) -> return,
`update_classifier`) — can be trained standalone and dropped in.

Why run DV-CG at all (runbook §2.5.7g): it is a PRE-REGISTERED test of the
demonstration-ceiling claim (results_chapter §7). Prediction: DV-CG lands
<= 75 on kitchen-mixed (it optimizes the same label-capped returns as every
other learned value). If it exceeded 75, the hard-wall interpretation would
be falsified — which is exactly what makes the run informative. It also
completes the guidance-resolution comparison for the write-up: trajectory-
level CG (this, on DV) vs token-level CG (mcts/noise_critic.py, on DF).

What this script does:
  1. builds the `_cg` results dir the pipeline's inference expects
     (same base-path template with guidance_type=cg);
  2. copies the MCSS planner/policy checkpoints into it (skipped if present);
  3. trains CumRewClassifier via the planner object's own `update_classifier`
     (guarantees the classifier sees the planner's exact noise schedule);
  4. tracks a held-out (path-level split) noisy-input correlation, saves
     `classifier_ckpt_best.pt`, and aliases BEST -> `classifier_ckpt_1000000.pt`
     — the name inference derives from planner_ckpt=1000000.

Then run the pipeline's own protocol (w needs tuning — the shipped
task.planner_w_cfg=1.0 is a placeholder, per the config's own comment):
  # cheap w-scan, 100 rollouts each (~40 min each):
  python pipelines/veteran_d4rl_kitchen.py mode=inference guidance_type=cg \
      enable_wandb=false num_episodes=2 task.planner_w_cfg=0.5
  (repeat: 1.0, 2.0, 4.0)
  # full protocol (1000 rollouts) at the best w:
  python pipelines/veteran_d4rl_kitchen.py mode=inference guidance_type=cg \
      enable_wandb=false task.planner_w_cfg=<best>

Run:
    python scripts/train_dv_classifier.py --smoke     # ~1 min gate
    python scripts/train_dv_classifier.py             # ~2-4 h
"""
import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from cleandiffuser.classifier import CumRewClassifier
from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.nn_classifier import HalfJannerUNet1d
from cleandiffuser.nn_diffusion import DiT1d
from mcts.specs import SPECS, env_family, make_dataset
from pipelines.utils import set_seed


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="kitchen-mixed-v0")
    p.add_argument("--steps", type=int, default=200000,
                   help="the pipeline trains the classifier for 1M steps "
                        "alongside the planner, but the analogous DV critic's "
                        "official inference ckpt is 200k (overfit beyond) — "
                        "same default here, with _best tracking as the guard")
    p.add_argument("--batch", type=int, default=128,
                   help="pipeline batch_size (configs/veteran/kitchen)")
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="fraction of PATHS held out (window split leaks)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-interval", type=int, default=50000)
    p.add_argument("--log-interval", type=int, default=2500)
    p.add_argument("--eval-interval", type=int, default=2500)
    p.add_argument("--smoke", action="store_true",
                   help="200 steps, no 1000000 alias: shapes/finiteness gate")
    p.add_argument("--src-ckpt", type=str, default=None,
                   help="MCSS ckpt dir holding planner/policy (default: "
                        "SPECS family dir)")
    p.add_argument("--out-dir", type=str, default=None,
                   help="the _cg dir the pipeline's cg inference reads "
                        "(default: src with _MCSS_ -> _cg_)")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    fam = env_family(args.env)
    if fam != "kitchen":
        sys.exit("DV-CG test is scoped to kitchen (the env where the DV paper "
                 "itself reports guidance beating MCSS)")
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    if args.smoke:
        args.steps, args.save_interval = 200, 10**9
        args.log_interval = args.eval_interval = 50

    src = Path((args.src_ckpt or SPECS[fam]["ckpt"])) / args.env
    if args.out_dir:
        out = Path(args.out_dir)
    else:
        if "_MCSS_" not in str(src):
            sys.exit(f"cannot derive the _cg dir: '_MCSS_' not in {src} — "
                     f"pass --out-dir explicitly")
        out = Path(str(src).replace("_MCSS_", "_cg_"))
    out.mkdir(parents=True, exist_ok=True)

    # the pipeline's cg inference loads planner_ckpt_1000000 + policy_ckpt_1000000
    # from the _cg dir — reuse the MCSS-trained ones (identical objective, see
    # module docstring). Copy, don't symlink (docker/Windows portability).
    for name in ("planner_ckpt_1000000.pt", "policy_ckpt_1000000.pt"):
        dst = out / name
        if dst.exists():
            print(f"  {name} already in {out} (kept)")
        elif (src / name).exists():
            shutil.copy2(src / name, dst)
            print(f"  copied {name}: {src} -> {out}")
        else:
            print(f"  WARNING: {src / name} missing — pipeline inference will "
                  f"fail until it exists")

    env, ds = make_dataset(args.env)
    H, stride, D = ds.horizon, ds.stride, ds.o_dim
    seq_obs = np.asarray(ds.seq_obs)
    seq_val = np.asarray(ds.seq_val)
    idx = np.asarray([(i[0], i[1]) for i in ds.indices], dtype=np.int64)
    print(f"[{args.env}] {len(idx):,} windows, H={H} stride={stride} D={D}")

    # ── the pipeline's exact cg construction (veteran_d4rl_kitchen.py):
    # HalfJannerUNet1d classifier inside ContinuousDiffusionSDE. The planner
    # net is built only so the SDE object exists — update_classifier uses the
    # SDE's add_noise schedule + the classifier, never the planner weights,
    # so no planner load is needed for training.
    nn_planner = DiT1d(D, emb_dim=128, d_model=256, n_heads=256 // 64,
                       depth=2, timestep_emb_type="fourier")
    nn_classifier = HalfJannerUNet1d(H, D, out_dim=1, model_dim=32, emb_dim=32,
                                     timestep_emb_type="positional",
                                     kernel_size=3)
    classifier = CumRewClassifier(nn_classifier, device=device)
    fix_mask = torch.zeros((H, D))
    fix_mask[0, :D] = 1.0
    loss_weight = torch.ones((H, D))
    planner = ContinuousDiffusionSDE(
        nn_planner, nn_condition=None, fix_mask=fix_mask,
        loss_weight=loss_weight, classifier=classifier, ema_rate=0.9999,
        device=device, predict_noise=True, noise_schedule="linear")
    sched = CosineAnnealingLR(classifier.optim, args.steps)

    # ── path-level train/val split (window split leaks; house rule) ─────────
    split_rng = np.random.default_rng(args.seed)
    paths = np.unique(idx[:, 0])
    perm = split_rng.permutation(len(paths))
    val_paths = set(paths[perm[:math.ceil(args.val_frac * len(paths))]].tolist())
    is_val = np.isin(idx[:, 0], list(val_paths))
    tr_idx, va_idx = idx[~is_val], idx[is_val]
    print(f"paths: total={len(paths)} val={len(val_paths)}  "
          f"windows: train={len(tr_idx)} val={len(va_idx)}")

    rng = np.random.default_rng(args.seed + 1)
    offs = np.arange(H) * stride

    def batch(pool, n):
        sel = pool[rng.integers(len(pool), size=n)]
        rows = sel[:, 1, None] + offs[None, :]
        x0 = torch.as_tensor(seq_obs[sel[:, 0, None], rows],
                             dtype=torch.float32, device=device)
        # (B, 1): the dataset's 'val' label (seq_val[path, start]) — the same
        # target the MCSS critic regresses (critic assert requires (B, 1))
        val = torch.as_tensor(seq_val[sel[:, 0], sel[:, 1]].reshape(-1, 1),
                              dtype=torch.float32, device=device)
        return x0, val

    # fixed val batch + ONE fixed noise draw (evals comparable across steps;
    # the eval must score actually-noised inputs at the SDE's own (xt, t))
    n_val = min(2048, len(va_idx))
    if n_val:
        vr = np.random.default_rng(args.seed + 2)
        vsel = va_idx[vr.choice(len(va_idx), size=n_val, replace=False)]
        vx = torch.as_tensor(seq_obs[vsel[:, 0, None],
                                     vsel[:, 1, None] + offs[None, :]],
                             dtype=torch.float32, device=device)
        vy = seq_val[vsel[:, 0], vsel[:, 1]].reshape(-1).astype(np.float32)
        torch.manual_seed(args.seed + 1234)
        vxt, vt, _ = planner.add_noise(vx)
        set_seed(args.seed)          # restore the global stream for training

        @torch.no_grad()
        def evaluate():
            noisy = classifier.logp(vxt, vt).squeeze(-1).cpu().numpy()
            clean = classifier.logp(vx, torch.zeros_like(vt)).squeeze(-1)
            return (pearson_corr(clean.cpu().numpy(), vy),
                    pearson_corr(noisy, vy))

    tag = "_smoke" if args.smoke else ""
    best_path = out / f"classifier_ckpt_best{tag}.pt"
    log, eval_log, best, best_step = [], [], -2.0, 0
    t0 = time.time()
    classifier.train()
    for step in range(1, args.steps + 1):
        x0, val = batch(tr_idx, args.batch)
        loss = planner.update_classifier(x0, val)["loss"]
        sched.step()
        if step % args.log_interval == 0 or step == args.steps:
            log.append(dict(step=step, loss=loss))
            print(f"step {step:>7}  loss={loss:.5f}  ({time.time() - t0:.0f}s)")
            if not np.isfinite(loss):
                sys.exit("ABORT: non-finite loss")
        if n_val and (step % args.eval_interval == 0 or step == args.steps):
            cc, nc = evaluate()
            eval_log.append(dict(step=step, clean_corr=cc, noisy_corr=nc))
            print(f"  eval {step:>7}  clean_corr={cc:+.3f}  noisy_corr={nc:+.3f}")
            if nc > best:
                best, best_step = nc, step
                classifier.save(str(best_path))
        if step % args.save_interval == 0:
            classifier.save(str(out / f"classifier_ckpt_{step}.pt"))

    classifier.eval()
    classifier.save(str(out / f"classifier_ckpt_final{tag}.pt"))
    if not args.smoke and best_path.exists():
        # inference derives the classifier name from planner_ckpt (=1000000):
        # alias BEST — the deployable checkpoint — to that name.
        shutil.copy2(best_path, out / "classifier_ckpt_1000000.pt")
        print(f"BEST noisy_corr={best:.3f} @ step {best_step} -> aliased to "
              f"classifier_ckpt_1000000.pt (what cg inference loads)")
    with open(out / f"classifier_train_log{tag}.json", "w") as f:
        json.dump(dict(env=args.env, args=vars(args), log=log,
                       eval_log=eval_log, best_step=best_step,
                       best_noisy_corr=best), f, indent=2)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
