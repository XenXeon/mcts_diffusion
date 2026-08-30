"""scripts/train_plan_value.py

Lever B step 2 — train the distilled plan-value  V-hat(s)  on the labels from
scripts/gen_plan_value_labels.py, and save it in the state-value checkpoint
format so it drops into the existing tree with ZERO loader changes:

    <ckpt dir>/<env>/state_value_ckpt_planv.pt
    -> scripts/run_mcts_compare.py ... --value-mode v_s --value-step planv

The experiment this settles: the old V(s) (same DVStateValue MLP, same MSE,
same states) plateaued at val corr ~0.74 on maze2d-large because its target
(behaviour return) is ill-posed. The plan-value target is a deterministic
function of the state (frozen planner+critic) plus iid sampling noise. If the
SAME architecture now reaches high val corr, the "bad network/training" story
is falsified and the target-posedness story confirmed.

Split is PATH-LEVEL (labels carry path ids) so val states never share a
trajectory with train states.

Run: python scripts/train_plan_value.py --env maze2d-large-v1
"""
import argparse
import json
import sys
import time

sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn.functional as F

from mcts.relabel import path_val_split
from mcts.specs import SPECS, env_family
from mcts.value_net import DVStateValue
from pipelines.utils import set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, required=True)
    p.add_argument("--labels", type=str, default=None,
                   help="npz from gen_plan_value_labels (default <ckpt dir>/plan_value_labels.npz)")
    p.add_argument("--target", type=str, default="topm",
                   choices=["topm", "max", "mean"],
                   help="which aggregate to regress (topm matches the tempered backup)")
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--eval-interval", type=int, default=2500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-tag", type=str, default="planv",
                   help="saved as state_value_ckpt_<tag>.pt (load: --value-step <tag>)")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    ckpt_dir = ((args.ckpt or SPECS[env_family(args.env)]["ckpt"])
                + f"/{args.env}")
    npz_path = args.labels or f"{ckpt_dir}/plan_value_labels.npz"
    z = np.load(npz_path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    if meta["env"] != args.env:
        sys.exit(f"label file is for {meta['env']}, not {args.env}")
    states = z["states"].astype(np.float32)
    labels = z[f"label_{args.target}"].astype(np.float32)[:, None]
    path_idx = z["path_idx"]
    obs_dim = states.shape[1]
    print(f"[{args.env}] {states.shape[0]:,} states, target={args.target} "
          f"(k={meta['k']}, top_m={meta['top_m']}), obs_dim={obs_dim}")

    # path-level split: a state's whole trajectory is either train or val
    n_paths = int(path_idx.max()) + 1
    val_paths, _ = path_val_split(n_paths, args.val_frac, args.seed)
    is_val = np.isin(path_idx, np.asarray(val_paths))
    tr_x, tr_y = states[~is_val], labels[~is_val]
    va_x = torch.tensor(states[is_val], device=device)
    va_y = torch.tensor(labels[is_val], device=device)
    print(f"train {tr_x.shape[0]:,} / val {int(is_val.sum()):,} states "
          f"({len(val_paths)} val paths)")

    net = DVStateValue(obs_dim, hidden_dim=args.hidden_dim, depth=args.depth,
                       dropout=args.dropout).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.steps)
    rng = np.random.default_rng(args.seed + 1)

    @torch.no_grad()
    def evaluate():
        net.eval()
        pred = net(va_x)
        mse = float(F.mse_loss(pred, va_y))
        a = pred.squeeze(-1).cpu().numpy()
        b = va_y.squeeze(-1).cpu().numpy()
        corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-8 else 0.0
        net.train()
        return mse, corr

    out_path = f"{ckpt_dir}/state_value_ckpt_{args.out_tag}.pt"
    best = (-2.0, -1)
    log = []
    t0 = time.time()
    net.train()
    for step in range(1, args.steps + 1):
        sel = rng.integers(tr_x.shape[0], size=args.batch)
        x = torch.tensor(tr_x[sel], device=device)
        y = torch.tensor(tr_y[sel], device=device)
        loss = F.mse_loss(net(x), y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        sched.step()
        if step % args.eval_interval == 0 or step == args.steps:
            mse, corr = evaluate()
            log.append(dict(step=step, val_mse=mse, val_corr=corr,
                            train_loss=float(loss.detach())))
            marker = ""
            if corr > best[0]:
                best = (corr, step)
                net.save(out_path, env=args.env, kind="plan_value",
                         target=args.target, label_meta=meta,
                         corr_val=corr, mse_val=mse, step=step)
                marker = "  <- best"
            print(f"step {step:>7}  val corr={corr:.4f} mse={mse:.5f} "
                  f"({time.time() - t0:.0f}s){marker}")

    with open(f"{ckpt_dir}/plan_value_train_log.json", "w") as f:
        json.dump(dict(env=args.env, args=vars(args), label_meta=meta,
                       best_corr=best[0], best_step=best[1], log=log), f, indent=2)
    print(f"saved BEST (corr {best[0]:.4f} @ step {best[1]}) -> {out_path}\n"
          f"deploy in the tree: --value-mode v_s --value-step {args.out_tag}\n"
          f"(old behaviour-return V(s) plateaued ~0.74 val corr on maze2d-large; "
          f"a large gap here = the target, not the net, was the problem)")


if __name__ == "__main__":
    main()
