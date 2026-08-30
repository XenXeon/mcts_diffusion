"""scripts/check_df_ckpt.py

Sample-quality gate for a Diffusion Forcing planner checkpoint — the check
that MATTERS (the eps-loss is not comparable to the DV planner's loss: DF is
causal + per-token noise on random-walk data, so it carries irreducible
conditional entropy the bidirectional DV objective never pays).

Prints, for n dataset start states:
  xy-hop     mean/p99 step size of generated windows vs real windows —
             gen ~ real and a sane p99 = trajectories are physically shaped;
  DV-critic  score distribution of generated vs real windows — same ballpark
             = the backbone is strong enough for closed-loop arms;
  prefix     conditional generation check: condition on the first j rows of
             a real window, verify the history rows come back untouched and
             the continuation's seam hop is dataset-like (this is the
             capability the whole DF arm exists for).

Run (safe while training continues — read-only):
    python scripts/check_df_ckpt.py --env maze2d-large-v1 --tag 100000
"""
import argparse
import sys

sys.path.insert(0, ".")

import numpy as np
import torch

from mcts.df_model import load_df_planner
from mcts.specs import SPECS, env_family, make_dataset
from pipelines.utils import set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, required=True)
    p.add_argument("--tag", type=str, default="final",
                   help="df_planner_ckpt_<tag>.pt")
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--prefix-j", type=int, default=8)
    p.add_argument("--row-stride", type=int, default=1)
    p.add_argument("--schedule", choices=["pyramid", "fullseq"], default="pyramid",
                   help="DF planner only: fullseq drops the causal-uncertainty "
                        "diagonal (the cheap-sampling quality gate)")
    p.add_argument("--sweeps", type=int, default=None,
                   help="shortcut planner only: sampling sweeps (power of 2)")
    p.add_argument("--cg-ckpt", type=str, default=None,
                   help="tag of a noise-aware value checkpoint "
                        "(noise_critic_ckpt_<tag>.pt, scripts/train_noise_critic.py) "
                        "to apply as classifier guidance on BOTH sample() calls "
                        "below, and to score real-vs-gen windows read-only")
    p.add_argument("--cg-w", type=float, default=0.0,
                   help="classifier-guidance weight passed to sample() (0 = "
                        "no-op by contract, even with --cg-ckpt set)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    ckpt_dir = ((args.ckpt or SPECS[env_family(args.env)]["ckpt"])
                + f"/{args.env}")
    planner = load_df_planner(f"{ckpt_dir}/df_planner_ckpt_{args.tag}.pt",
                              device=device)
    # Optional classifier-guidance critic (mcts/noise_critic.py) — read-only here:
    # this script never trains anything, it just threads guide=/w_cg= through the
    # two sample() calls below and reports whether guidance moved the generations.
    cg_critic = None
    if args.cg_ckpt:
        from mcts.noise_critic import NoiseAwareCritic
        cg_critic = NoiseAwareCritic.load(
            f"{ckpt_dir}/noise_critic_ckpt_{args.cg_ckpt}.pt", device=device)
        print(f"  loaded noise-critic: noise_critic_ckpt_{args.cg_ckpt}.pt "
              f"(cfg={cg_critic.cfg})")
        # Same silent-no-op refusal as Sampler.__init__ (mcts_loop.py): the
        # shortcut planner's sample() absorbs guide=/w_cg= via **_ without
        # applying them, so a "guided" gate of a shortcut checkpoint would
        # print unguided numbers under a guided header. Refuse instead.
        if planner.cfg.get("kind") == "shortcut":
            sys.exit("--cg-ckpt with a shortcut planner checkpoint: the "
                     "shortcut sampler has no guidance hook — gate CG on the "
                     "standard DF planner (e.g. --tag final)")
        if cg_critic.cfg["K"] != planner.cfg["K"]:
            sys.exit(f"noise-critic K={cg_critic.cfg['K']} != DF planner "
                     f"K={planner.cfg['K']} — alpha-bar tables must match")
    env, ds = make_dataset(args.env)
    H, stride, D = ds.horizon, ds.stride, ds.o_dim
    seq_obs = np.asarray(ds.seq_obs)
    idx = np.asarray([(i[0], i[1]) for i in ds.indices], dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    sel = idx[rng.integers(len(idx), size=args.n)]
    offs = np.arange(H) * stride
    real = torch.as_tensor(seq_obs[sel[:, 0, None], sel[:, 1, None] + offs],
                           dtype=torch.float32, device=device)

    hop = lambda w: (w[:, 1:, :2] - w[:, :-1, :2]).norm(dim=-1)
    p99 = lambda v: float(v.flatten().kthvalue(
        max(1, int(0.99 * v.numel())))[0])

    # 1) generation from s0 only (the MCSS setting)
    x_hist = torch.zeros_like(real)
    x_hist[:, 0] = real[:, 0]
    gen = planner.sample(x_hist, torch.ones(args.n, dtype=torch.long), H,
                         schedule=args.schedule, row_stride=args.row_stride,
                         sweeps=args.sweeps, guide=cg_critic, w_cg=args.cg_w)
    print(f"[{args.env} @ {args.tag}] n={args.n}, schedule={args.schedule}, "
          f"row_stride={args.row_stride}, sweeps={args.sweeps}, cg_w={args.cg_w}")
    print(f"xy-hop   real mean {hop(real).mean():.4f} p99 {p99(hop(real)):.4f}"
          f"  |  gen mean {hop(gen).mean():.4f} p99 {p99(hop(gen)):.4f}")

    # noise-critic's own CLEAN (k=0) score of real vs generated windows — read-only
    # diagnostic: shows whether guidance (--cg-w != 0) actually pushed generations
    # toward higher predicted value. guide=cg_critic, w_cg=0 above is a no-op by
    # contract, so this print is informative even with --cg-w 0 (baseline gap).
    if cg_critic is not None:
        try:
            k0 = torch.zeros(args.n, H, dtype=torch.long, device=device)
            with torch.no_grad():
                nc_real = cg_critic.value(real, k0)
                nc_gen = cg_critic.value(gen, k0)
            print(f"noise-critic (clean)  real {nc_real.mean():.4f}±{nc_real.std():.4f}"
                  f"  |  gen {nc_gen.mean():.4f}±{nc_gen.std():.4f}  "
                  f"(higher gen = guidance pushed toward higher predicted value)")
        except Exception as exc:
            print(f"(noise-critic score skipped: {exc!r})")

    # 2) DV critic ballpark
    try:
        from cleandiffuser.utils import DVHorizonCritic
        critic = DVHorizonCritic(D, emb_dim=128, d_model=256, n_heads=4,
                                 depth=2, norm_type="pre").to(device)
        critic.load_state_dict(torch.load(
            f"{ckpt_dir}/critic_ckpt_1000000.pt", map_location=device,
            weights_only=False)["critic"])
        critic.eval()
        with torch.no_grad():
            sr, sg = critic(real).squeeze(-1), critic(gen).squeeze(-1)
        print(f"DV-critic  real {sr.mean():.4f}±{sr.std():.4f}  |  "
              f"gen {sg.mean():.4f}±{sg.std():.4f}")
    except Exception as exc:
        print(f"(critic check skipped: {exc!r})")

    # 3) prefix-conditioned continuation (the tree-search capability)
    j = args.prefix_j
    x_hist = real.clone()          # rows [0:j] = real history, rest ignored
    hl = torch.full((args.n,), j, dtype=torch.long)
    cond = planner.sample(x_hist, hl, H, schedule=args.schedule,
                          row_stride=args.row_stride, sweeps=args.sweeps,
                          guide=cg_critic, w_cg=args.cg_w)
    hist_err = float((cond[:, :j] - real[:, :j]).abs().max())
    seam = (cond[:, j, :2] - cond[:, j - 1, :2]).norm(dim=-1)
    cont_hop = hop(cond[:, j:])
    print(f"prefix j={j}: hist_err {hist_err:.2e} (must be ~0)  seam mean "
          f"{seam.mean():.4f}  cont-hop mean {cont_hop.mean():.4f} "
          f"p99 {p99(cont_hop):.4f}")
    print("READ: gen hops ~ real hops, critic same ballpark, hist_err ~0, "
          "seam ~ hop scale -> backbone ready for closed-loop "
          "(run_mcts_compare --df-ckpt).")


if __name__ == "__main__":
    main()
