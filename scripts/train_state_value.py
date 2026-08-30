"""scripts/train_state_value.py

Train the MCTS state-value critic  V(s) -> normalised return-to-go.

This is the retrained critic for the MCTS-as-sampler integration (Option 1: state-value
+ segment stitching).  It reuses DV's already-trained planner and inverse-dynamics policy
untouched; only this value head is new.

Supervision (identical distribution to DV's MCSS critic, just per-state)
-----------------------------------------------------------------------
The DV dataset already computes a per-timestep discounted return-to-go `seq_val[p, t]`
and stores GaussianNormalizer-normalised states `seq_obs[p, t]`.  The MCSS critic trains
on ( whole trajectory , seq_val[p, start] ); we train on ( seq_obs[p, start] ,
seq_val[p, start] ) for the same valid start indices (dataset.indices).  So the only
change is the input representation: a single state instead of the full plan.

The target config MUST match the training pipeline exactly or V is not comparable to the
MCSS critic.  From configs/veteran/*/reward_mode/linear.yaml and the pipeline:
    discount=1.0, continous_reward_at_done=True, reward_tune="iql",
    center_mapping=True (because guidance_type=MCSS != cfg).
With these, seq_val = normalised NEGATIVE-time-to-goal in [-1, 1]  (1 == at goal).

Run:
    python scripts/train_state_value.py --env maze2d-large-v1
    python scripts/train_state_value.py --env antmaze-large-diverse-v2 --steps 200000

Output:
    <planner ckpt dir>/<env>/state_value_ckpt_{step}.pt   (co-located with planner/critic)
    <planner ckpt dir>/<env>/state_value_train_log.json
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

from mcts.relabel import build_relabel_inputs, path_val_split, sample_batch
from mcts.specs import SPECS, TARGET_CFG, env_family, make_dataset
from mcts.value_net import DVStateValue, DVStateValueEnsemble
from mcts.value_scale import StepScale
from pipelines.utils import set_seed


def expectile_loss(pred: torch.Tensor, target: torch.Tensor,
                   tau: float) -> torch.Tensor:
    """Asymmetric L2: weight tau above the prediction, (1-tau) below.

    On exact MC relabelled targets this approximates the min-time over
    in-support behaviour (R5.2) — MSE would converge to the MEAN wandering
    time, biasing exactly the child-ranking the tree performs.
    """
    u = target - pred
    w = torch.where(u < 0, 1.0 - tau, tau)
    return (w * u * u).mean()


def train_goal_conditioned(args, ds, device, save_dir):
    """Plan v5.1 §3a: relabelled V(s, g) ensemble on the shared pipeline affine."""
    if not hasattr(ds, "seq_tml"):
        sys.exit("--goal-conditioned requires a dataset with seq_tml — the "
                 "spatial-goal envs antmaze and maze2d (obs[:, :2] = xy). "
                 "kitchen has no spatial goal (obs[:, :2] are joint angles; the "
                 "goal is a subtask set), so V(s, g) does not apply there.")

    seq_val = np.asarray(ds.seq_val)
    obs_dim = ds.o_dim
    # Shared derivation (single source — also used by D1/D4): seq_obs, the
    # per-path goal-caps `terminus` (incl. timeout last-state in full-data), the
    # terminus-only `term_only` that anchors D=867, and the StepScale.
    seq_obs, terminus, term_only, scale = build_relabel_inputs(ds)
    n_term, n_timeout = len(term_only), len(terminus) - len(term_only)
    if args.full_data:
        print(f"FULL-DATA mode: {n_term} terminus + {n_timeout} timeout paths "
              f"({100*n_timeout/len(terminus):.0f}% extra). D={scale.D} anchored on "
              f"terminus paths; seq_val consistency assert SKIPPED (the full dataset "
              f"re-normalises seq_val, which relabeling does not use — targets come "
              f"from offsets on the fixed D scale).")
    else:
        # R5.7a: the relabelled targets must sit on the dataset's own affine. Verify
        # against seq_val at several path starts; abort loudly on any mismatch.
        for p_idx in range(0, len(terminus), max(1, len(terminus) // 7)):
            scale.assert_consistent_with_dataset(
                float(seq_val[p_idx, 0]), terminus[p_idx], tol=1e-3)
        print(f"value scale verified: D={scale.D} dense steps; "
              f"target(t'=t)=1.0 == point-to-segment terminal value")

    # Path-level train/val split — shared with D4 so 'held-out' is the same paths.
    val_paths, tr_paths = path_val_split(seq_obs.shape[0], args.val_frac, args.seed)
    print(f"paths: train={len(tr_paths)} val={len(val_paths)}  "
          f"mixture 70/20/10 future/terminus/current, geo_mean={args.geo_mean}")

    # Fixed validation set (deterministic; D4 reuses the same construction).
    vs, vg, vt = sample_batch(seq_obs, terminus, scale, args.val_pairs,
                              args.geo_mean, np.random.default_rng(args.seed + 1),
                              paths=val_paths)
    val_x = torch.tensor(np.concatenate([vs, vg], axis=-1), device=device)
    val_t = torch.tensor(vt, device=device)

    net = DVStateValueEnsemble(obs_dim + 2, n_members=args.ensemble,
                               hidden_dim=args.hidden_dim, depth=args.depth,
                               dropout=args.dropout,
                               goal_conditioned=True).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"DVStateValueEnsemble: in_dim={obs_dim + 2} members={args.ensemble} "
          f"loss={args.loss}(tau={args.tau}) params={n_params:,}")
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.steps)
    # Independent batch streams per member — ensemble diversity via data, not init alone.
    member_rngs = [np.random.default_rng(args.seed + 100 + i)
                   for i in range(args.ensemble)]

    def loss_fn(pred, target):
        if args.loss == "expectile":
            return expectile_loss(pred, target, args.tau)
        return F.mse_loss(pred, target)

    @torch.no_grad()
    def evaluate():
        net.eval()
        v = net(val_x)                                       # (B, N)
        per_member = [float(loss_fn(v[:, i:i + 1], val_t)) for i in range(v.shape[1])]
        vmin = v.min(dim=1, keepdim=True).values
        a = v.mean(dim=1).cpu().numpy()
        b = val_t.squeeze(-1).cpu().numpy()
        corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-8 else 0.0
        # calibration in STEPS via the shared affine (C2) — the unit D4 reports in
        steps_err = float(np.abs(scale.steps(vmin.squeeze(-1).cpu().numpy())
                                 - scale.steps(b)).mean())
        net.train()
        return float(np.mean(per_member)), corr, steps_err

    # Checkpoint prefix encodes the data regime so terminus-only and full-data
    # critics coexist for the data-efficiency ablation (11% vs 100% of trajectories).
    prefix = "state_value_sg_full" if args.full_data else "state_value_sg"
    # val_frac/seed persisted so D4 reproduces the EXACT held-out split (R-C):
    # the split is (num_paths, val_frac, seed); auto-detect already matches
    # num_paths via the dataset regime, these close the remaining two knobs.
    meta = dict(env=args.env, center_mapping=TARGET_CFG["center_mapping"],
                D=scale.D, geo_mean=args.geo_mean, loss=args.loss, tau=args.tau,
                full_data=args.full_data, n_term=n_term, n_timeout=n_timeout,
                val_frac=args.val_frac, seed=args.seed)
    # NOTE: val_loss is on the training loss scale and is NOT comparable across
    # the MSE-vs-expectile ablation (different scales). Select/compare ablation
    # arms by val_corr and val_min_steps_mae (loss-independent), per §3a / §9.
    log = {"step": [], "train_loss": [], "val_loss": [], "val_corr": [],
           "val_min_steps_mae": []}
    best_corr = -2.0                # best-val checkpoint: this critic overfits, so
    best_step = 0                   # _best.pt (peak val_corr) is the one to deploy
    t0 = time.perf_counter()
    net.train()
    run_loss = 0.0
    for step in range(1, args.steps + 1):
        total = 0.0
        optim.zero_grad(set_to_none=True)
        for i, member in enumerate(net.members):
            s, g, t = sample_batch(seq_obs, terminus, scale, args.batch_size,
                                   args.geo_mean, member_rngs[i], paths=tr_paths)
            x = torch.tensor(np.concatenate([s, g], axis=-1), device=device)
            tt = torch.tensor(t, device=device)
            loss = loss_fn(member(x), tt)
            loss.backward()
            total += float(loss)
        optim.step()
        sched.step()
        run_loss += total / args.ensemble

        if step % args.log_interval == 0:
            train_loss = run_loss / args.log_interval
            run_loss = 0.0
            val_loss, val_corr, steps_mae = evaluate()
            dt = time.perf_counter() - t0
            print(f"step {step:>7}/{args.steps}  train={train_loss:.4f}  "
                  f"val={val_loss:.4f}  corr={val_corr:.3f}  "
                  f"min-steps-MAE={steps_mae:.0f}  ({step/dt:.0f} it/s)")
            log["step"].append(step)
            log["train_loss"].append(train_loss)
            log["val_loss"].append(val_loss)
            log["val_corr"].append(val_corr)
            log["val_min_steps_mae"].append(steps_mae)
            # best-val checkpoint (peak val_corr) — the deploy target, since the
            # ensemble overfits well before --steps (val_corr peaks then declines).
            if val_corr > best_corr:
                best_corr, best_step = val_corr, step
                net.save(f"{save_dir}/{prefix}_ckpt_best.pt", best_step=step,
                         best_val_corr=val_corr, **meta)

        if step % args.save_interval == 0 or step == args.steps:
            net.save(f"{save_dir}/{prefix}_ckpt_{step}.pt", **meta)
            net.save(f"{save_dir}/{prefix}_ckpt_latest.pt", **meta)

    with open(f"{save_dir}/{prefix}_train_log.json", "w") as f:
        json.dump({"args": vars(args), "target_cfg": TARGET_CFG, "meta": meta,
                   "log": log, "best_step": best_step, "best_val_corr": best_corr},
                  f, indent=2)
    val_loss, val_corr, steps_mae = evaluate()
    print(f"\nDONE  final val_corr={val_corr:.3f} (steps-MAE={steps_mae:.0f})  |  "
          f"BEST val_corr={best_corr:.3f} @ step {best_step} -> {prefix}_ckpt_best.pt")
    print(f"Run D1 on the BEST checkpoint, not latest (overfit):\n"
          f"  python scripts/diag_d1_compass.py --env {args.env} "
          f"--sg-ckpt {prefix}_ckpt_best.pt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, required=True)
    p.add_argument("--steps", type=int, default=200000,
                   help="gradient steps (matches MCSS critic's 200k checkpoint)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--val-frac", type=float, default=0.05,
                   help="fraction of PATHS held out for validation (path-level split)")
    p.add_argument("--save-interval", type=int, default=50000)
    p.add_argument("--log-interval", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--ckpt", type=str, default=None, help="override save dir base")
    # ── Goal-conditioned V(s, g) mode (plan v5.1 §3a) ──────────────────────────
    p.add_argument("--goal-conditioned", action="store_true",
                   help="train the relabelled V(s, g) ensemble instead of V(s)")
    p.add_argument("--full-data", action="store_true",
                   help="relabel from ALL trajectories incl. timeout paths "
                        "(learn_policy=True; ~89%% more data on antmaze-large) — "
                        "fixes the 11%%-of-data starvation, expands the manifold")
    p.add_argument("--ensemble", type=int, default=5,
                   help="ensemble members (goal-conditioned mode)")
    p.add_argument("--loss", choices=["expectile", "mse"], default="expectile",
                   help="expectile=headline (R5.2: min-time over in-support "
                        "behaviour); mse=pre-registered ablation arm")
    p.add_argument("--tau", type=float, default=0.9,
                   help="expectile level (ablation list: 0.7, 0.9)")
    p.add_argument("--geo-mean", type=float, default=200.0,
                   help="mean of the geometric future-goal offset (dense steps)")
    p.add_argument("--val-pairs", type=int, default=20000,
                   help="fixed relabelled validation pairs from held-out paths")
    args = p.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    fam = env_family(args.env)
    SPEC = SPECS[fam]
    H, stride = SPEC["H"], SPEC["stride"]
    save_dir = (args.ckpt or SPEC["ckpt"]) + f"/{args.env}"
    os.makedirs(save_dir, exist_ok=True)

    print(f"[{args.env}] family={fam} H={H} stride={stride} device={device}")
    print(f"target cfg: {TARGET_CFG}")
    # Full-data relabeling needs timeout paths (learn_policy=True); V(s) and the
    # terminus-only V(s,g) keep the default terminus-reaching set.
    env, ds = make_dataset(args.env, H=H, stride=stride,
                           learn_policy=(args.goal_conditioned and args.full_data))

    if args.goal_conditioned:
        return train_goal_conditioned(args, ds, device, save_dir)

    obs_dim = ds.o_dim
    seq_obs = np.asarray(ds.seq_obs)   # (P, T, obs_dim) — normalised states
    seq_val = np.asarray(ds.seq_val)   # (P, T, 1)       — normalised return-to-go
    print(f"seq_obs {seq_obs.shape}  seq_val {seq_val.shape}")
    print(f"value target range: min={seq_val.min():.3f} max={seq_val.max():.3f} "
          f"mean={seq_val.mean():.3f}  (expect [-1,1], 1==at goal)")

    # ── (state, value) pairs from valid start indices, path-level train/val split ──
    pairs = np.array([(pp, ss) for (pp, ss, _ee) in ds.indices], dtype=np.int64)  # (N, 2)
    num_paths = seq_obs.shape[0]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(num_paths)
    n_val = max(1, int(args.val_frac * num_paths))
    val_paths = set(perm[:n_val].tolist())
    is_val = np.array([pp in val_paths for pp in pairs[:, 0]])
    tr, va = pairs[~is_val], pairs[is_val]
    print(f"pairs: total={len(pairs)}  train={len(tr)}  val={len(va)}  "
          f"({num_paths} paths, {n_val} held out)")

    def gather(idx_arr):
        s = torch.tensor(seq_obs[idx_arr[:, 0], idx_arr[:, 1]], dtype=torch.float32)
        v = torch.tensor(seq_val[idx_arr[:, 0], idx_arr[:, 1]], dtype=torch.float32)
        if v.ndim == 1:
            v = v.unsqueeze(-1)
        return s, v

    tr_s, tr_v = gather(tr)
    va_s, va_v = gather(va)
    va_s, va_v = va_s.to(device), va_v.to(device)
    N = tr_s.shape[0]

    # ── Model / optim ──────────────────────────────────────────────────────────
    net = DVStateValue(obs_dim, hidden_dim=args.hidden_dim, depth=args.depth,
                       dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"DVStateValue: obs_dim={obs_dim} hidden={args.hidden_dim} depth={args.depth} "
          f"params={n_params:,}")
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.steps)

    use_expectile = args.loss == "expectile"
    if use_expectile:
        print(f"V(s) loss: expectile(tau={args.tau}) — biases towards the best "
              f"(shortest-path) return-to-go, reducing behaviour-policy wandering noise")
    else:
        print(f"V(s) loss: MSE — learns the mean behaviour-policy return-to-go")

    def train_loss(pred, target):
        if use_expectile:
            return expectile_loss(pred, target, args.tau)
        return F.mse_loss(pred, target)

    @torch.no_grad()
    def evaluate():
        net.eval()
        pred = net(va_s)
        mse = F.mse_loss(pred, va_v).item()
        a = pred.squeeze(-1).cpu().numpy()
        b = va_v.squeeze(-1).cpu().numpy()
        corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-8 else 0.0
        net.train()
        return mse, corr

    log = {"step": [], "train_mse": [], "val_mse": [], "val_corr": []}
    # V(s) OVERFITS: on antmaze val_corr peaks ~0.87 near step 6k then declines to ~0.81 by
    # 1M as train_mse->0. save_interval (50k) is too coarse to catch that early peak, so track
    # the best val_corr and save state_value_ckpt_best.pt — deploy that (not _latest).
    best_corr, best_step = -2.0, 0
    t0 = time.perf_counter()
    net.train()
    run_loss = 0.0
    for step in range(1, args.steps + 1):
        bidx = torch.randint(0, N, (args.batch_size,))
        s = tr_s[bidx].to(device)
        v = tr_v[bidx].to(device)
        pred = net(s)
        loss = train_loss(pred, v)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        sched.step()
        run_loss += loss.item()

        if step % args.log_interval == 0:
            train_mse = run_loss / args.log_interval
            run_loss = 0.0
            val_mse, val_corr = evaluate()
            dt = time.perf_counter() - t0
            print(f"step {step:>7}/{args.steps}  train_mse={train_mse:.4f}  "
                  f"val_mse={val_mse:.4f}  val_corr={val_corr:.3f}  "
                  f"({step/dt:.0f} it/s)")
            log["step"].append(step)
            log["train_mse"].append(train_mse)
            log["val_mse"].append(val_mse)
            log["val_corr"].append(val_corr)
            if val_corr > best_corr:            # peak-val_corr checkpoint = the deploy target
                best_corr, best_step = val_corr, step
                net.save(f"{save_dir}/state_value_ckpt_best.pt",
                         center_mapping=TARGET_CFG["center_mapping"], env=args.env,
                         best_step=step, best_val_corr=val_corr,
                         loss=args.loss, tau=args.tau)

        if step % args.save_interval == 0 or step == args.steps:
            net.save(f"{save_dir}/state_value_ckpt_{step}.pt",
                     center_mapping=TARGET_CFG["center_mapping"], env=args.env,
                     loss=args.loss, tau=args.tau)
            net.save(f"{save_dir}/state_value_ckpt_latest.pt",
                     center_mapping=TARGET_CFG["center_mapping"], env=args.env,
                     loss=args.loss, tau=args.tau)

    with open(f"{save_dir}/state_value_train_log.json", "w") as f:
        json.dump({"args": vars(args), "target_cfg": TARGET_CFG, "log": log,
                   "best_step": best_step, "best_val_corr": best_corr}, f, indent=2)

    final_mse, final_corr = evaluate()
    print(f"\nDONE  val_mse={final_mse:.4f}  final val_corr={final_corr:.3f}  |  "
          f"BEST val_corr={best_corr:.3f} @ step {best_step} -> state_value_ckpt_best.pt")
    print(f"Deploy the BEST ckpt — V(s) overfits, so pass --value-step best to "
          f"run_mcts_compare (NOT the default 'latest', which is past the peak).")
    print(f"Reference peaks: antmaze-large-diverse 0.874, maze2d-large 0.76 (label noise).")


if __name__ == "__main__":
    main()
