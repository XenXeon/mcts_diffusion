"""scripts/gen_plan_value_labels.py

Lever B step 1 — generate DISTILLED PLAN-VALUE labels:  V-hat(s) targets.

The old V(s) regressed the BEHAVIOUR-policy return from s — an ill-posed target
(same state, many futures/goals -> SNR ceiling; val_corr plateaued ~0.74 on
maze2d-large no matter the loss/arch). The plan-value target is different:

    label(s) = aggregate over K planner samples from s of critic(trajectory)

i.e. "what would MCSS score from here". Given the FROZEN planner+critic, the
conditional expectation of this label given s is a DETERMINISTIC function of s;
the only label noise is iid sampling noise, which MSE regression averages out.
Well-posed where the behaviour-return target was not — if the SAME MLP now
reaches high val corr, that falsifies "the V(s) net/training was the problem"
and confirms the target was.

Aggregates stored per state (choose at training time): max over K (the literal
MCSS outcome, winner's-curse-inflated), mean, and mean of the top-m (m=3 by
default — matches the tempered backup the tree runs validated).

States are drawn from ds.indices start-states — the exact supervision support
of the old V(s) (mcts/value_net.py docstring), so the two are comparable.

Run (GPU box; ~30-60 min at the defaults):
    python scripts/gen_plan_value_labels.py --env maze2d-large-v1

Output: <ckpt dir>/<env>/plan_value_labels.npz
"""
import argparse
import json
import sys
import time

sys.path.insert(0, ".")

import numpy as np
import torch

from mcts.mcts_loop import load_models
from mcts.specs import make_dataset
from pipelines.utils import set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, required=True)
    p.add_argument("--n-states", type=int, default=100000)
    p.add_argument("--k", type=int, default=16, help="planner samples per state")
    p.add_argument("--top-m", type=int, default=3)
    p.add_argument("--batch-states", type=int, default=32,
                   help="states per planner call (x k trajectories)")
    p.add_argument("--plan-steps", type=int, default=20,
                   help="diffusion sample steps — match inference (Sampler default)")
    p.add_argument("--solver", type=str, default="ddim")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--critic-step", type=str, default="1000000")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default=None,
                   help="output npz (default <ckpt dir>/plan_value_labels.npz)")
    args = p.parse_args()

    set_seed(args.seed)
    m = load_models(args.env, critic_step=args.critic_step, device=args.device,
                    ckpt_dir=args.ckpt)
    dev, H, D = m["device"], m["H"], m["obs_dim"]
    planner, critic = m["planner"], m["critic"]

    _, ds = make_dataset(args.env)
    starts = list(ds.indices)                       # (path, start, end) rows
    rng = np.random.default_rng(args.seed)
    picks = rng.permutation(len(starts))[:args.n_states]
    seq_obs = np.asarray(ds.seq_obs)
    path_idx = np.array([starts[i][0] for i in picks], dtype=np.int64)
    t_idx = np.array([starts[i][1] for i in picks], dtype=np.int64)
    states = seq_obs[path_idx, t_idx].astype(np.float32)   # (N, D) normalised
    n = states.shape[0]
    print(f"[{args.env}] labelling {n:,} start-states  "
          f"(k={args.k}, top_m={args.top_m}, plan_steps={args.plan_steps})")

    S, K = args.batch_states, args.k
    lab_max = np.empty(n, dtype=np.float32)
    lab_mean = np.empty(n, dtype=np.float32)
    lab_topm = np.empty(n, dtype=np.float32)
    t0 = time.time()
    for lo in range(0, n, S):
        hi = min(lo + S, n)
        b = hi - lo
        s = torch.tensor(states[lo:hi], device=dev)
        prior = torch.zeros((b * K, H, D), device=dev)
        prior[:, 0, :] = s.repeat_interleave(K, dim=0)
        with torch.no_grad():
            trajs, _ = planner.sample(
                prior, solver=args.solver, n_samples=b * K,
                sample_steps=args.plan_steps, use_ema=True,
                condition_cfg=None, w_cfg=1.0, temperature=1.0)
            scores = critic(trajs).squeeze(-1).view(b, K)      # (b, K)
        sc = scores.cpu().numpy()
        lab_max[lo:hi] = sc.max(axis=1)
        lab_mean[lo:hi] = sc.mean(axis=1)
        mm = min(args.top_m, K)
        lab_topm[lo:hi] = np.sort(sc, axis=1)[:, -mm:].mean(axis=1)
        if (lo // S) % 50 == 0:
            done = hi / n
            eta = (time.time() - t0) / max(done, 1e-9) * (1 - done)
            print(f"  {hi:>7}/{n}  ({100 * done:.1f}%)  eta {eta / 60:.0f} min")

    out = args.out or f"{m['ckpt_dir']}/plan_value_labels.npz"
    meta = dict(env=args.env, n=n, k=args.k, top_m=args.top_m,
                plan_steps=args.plan_steps, solver=args.solver,
                critic_step=str(args.critic_step), seed=args.seed,
                planner_step=m["planner_step"], obs_dim=D, H=H)
    np.savez_compressed(out, states=states, path_idx=path_idx, t_idx=t_idx,
                        label_max=lab_max, label_mean=lab_mean,
                        label_topm=lab_topm, meta=json.dumps(meta))
    print(f"saved {out}  ({time.time() - t0:.0f}s)\n"
          f"  label ranges: max [{lab_max.min():.3f}, {lab_max.max():.3f}]  "
          f"topm [{lab_topm.min():.3f}, {lab_topm.max():.3f}]")


if __name__ == "__main__":
    main()
