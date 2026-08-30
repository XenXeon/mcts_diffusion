"""scripts/check_grounded_pool.py

THE GO/NO-GO DIAGNOSTIC for the grounded-tree / grounded-MCSS moonshot
(mcts/grounded.py) — read-only, open-loop, spends NO closed-loop GPU.

Every LEARNED value in this stack (DV critic, V(s), V(s,g), noise-critic) is
trained on kitchen-mixed labels capped at 3-of-4 subtasks: no demonstration in
the dataset ever solves all 4. Before spending closed-loop GPU on a grounded
tree / MCSS arm (mcts_loop.py value_mode="grounded" / --grounded-mcss), this
script answers the prior question the grounded value's whole premise depends
on: can the frozen DF planner even IMAGINE the (min_solved+1)-th subtask
completion when conditioned on a real dataset state that already has
min_solved subtasks done? The dataset never demonstrates that continuation,
so any sampled window reaching it is pure generalization beyond the training
support — if the planner NEVER produces one across many samples from many
such states, the grounded value has nothing to select for at the sampling
level and the moonshot dies here, cheaply, before any closed-loop run.

Run (safe while training continues — read-only):
    python scripts/check_grounded_pool.py --env kitchen-mixed-v0 --tag final
"""
import argparse
import sys

sys.path.insert(0, ".")

import numpy as np
import torch

from mcts.df_model import load_df_planner
from mcts.grounded import KitchenGroundedChecker
from mcts.specs import SPECS, env_family, make_dataset
from pipelines.utils import set_seed

# Selected-state windows are sampled in chunks of this many states at once
# (each chunk draws chunk_size * --k windows) to bound memory — k*n = 9600
# windows of (32, 60) is small overall, but chunking keeps larger --n/--k
# requests safe too.
STATE_CHUNK = 8


def _chunked_count(checker: KitchenGroundedChecker, states_norm: np.ndarray,
                   device: str, chunk: int = 4096) -> np.ndarray:
    """(N, D) normalized states -> (N,) float64 grounded solved-counts, scored as
    T=1 windows (checker.count union-over-t degenerates to "solved at this one
    state" when T=1). Batched because kitchen-mixed has tens of thousands of
    dataset window-start states."""
    out = np.empty(states_norm.shape[0], dtype=np.float64)
    for i in range(0, states_norm.shape[0], chunk):
        blk = states_norm[i:i + chunk]
        x = torch.as_tensor(blk[:, None, :], dtype=torch.float32, device=device)
        out[i:i + chunk] = checker.count(x).cpu().numpy()
    return out


def _print_dist(name: str, counts: np.ndarray) -> None:
    counts = np.asarray(counts)
    vals, freq = np.unique(np.round(counts).astype(np.int64), return_counts=True)
    dist = ", ".join(f"{v}:{f}" for v, f in zip(vals, freq))
    print(f"  {name}  n={counts.size}  mean={counts.mean():.3f}  [{dist}]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="kitchen-mixed-v0")
    p.add_argument("--tag", type=str, default="final",
                   help="df_planner_ckpt_<tag>.pt")
    p.add_argument("--min-solved", type=int, default=3,
                   help="condition on dataset states with exactly this many "
                        "grounded-solved subtasks (3 = the dataset's label cap)")
    p.add_argument("--n", type=int, default=64,
                   help="number of dataset conditioning states")
    p.add_argument("--k", type=int, default=150,
                   help="sampled windows per conditioning state")
    p.add_argument("--cg-ckpt", type=str, default=None,
                   help="tag of a noise-aware value checkpoint "
                        "(noise_critic_ckpt_<tag>.pt, scripts/train_noise_critic.py) "
                        "to apply as classifier guidance on sampling — read-only "
                        "diagnostic use, mirrors scripts/check_df_ckpt.py")
    p.add_argument("--cg-w", type=float, default=0.0,
                   help="classifier-guidance weight passed to sample() (0 = "
                        "no-op by contract, even with --cg-ckpt set)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    if env_family(args.env) != "kitchen":
        sys.exit(f"check_grounded_pool needs a kitchen env (got {args.env!r}, "
                 f"family={env_family(args.env)!r}) — the grounded checker "
                 f"reads kitchen task definitions off the live env")
    ckpt_dir = (args.ckpt or SPECS[env_family(args.env)]["ckpt"]) + f"/{args.env}"
    planner = load_df_planner(f"{ckpt_dir}/df_planner_ckpt_{args.tag}.pt",
                              device=device)

    # Optional classifier-guidance critic (mcts/noise_critic.py) — read-only here,
    # same silent-no-op refusal as check_df_ckpt.py / Sampler.__init__: the
    # shortcut planner's sample() absorbs guide=/w_cg= via **_ without applying
    # them, so a "guided" gate of a shortcut checkpoint would print unguided
    # numbers under a guided header. Refuse instead.
    cg_critic = None
    if args.cg_ckpt:
        from mcts.noise_critic import NoiseAwareCritic
        cg_critic = NoiseAwareCritic.load(
            f"{ckpt_dir}/noise_critic_ckpt_{args.cg_ckpt}.pt", device=device)
        print(f"  loaded noise-critic: noise_critic_ckpt_{args.cg_ckpt}.pt "
              f"(cfg={cg_critic.cfg})")
        if planner.cfg.get("kind") == "shortcut":
            sys.exit("--cg-ckpt with a shortcut planner checkpoint: the "
                     "shortcut sampler has no guidance hook — gate CG on the "
                     "standard DF planner (e.g. --tag final)")
        if cg_critic.cfg["K"] != planner.cfg["K"]:
            sys.exit(f"noise-critic K={cg_critic.cfg['K']} != DF planner "
                     f"K={planner.cfg['K']} — alpha-bar tables must match")

    env, ds = make_dataset(args.env)
    H, D = ds.horizon, ds.o_dim
    checker = KitchenGroundedChecker.from_env(env, ds.get_normalizer())

    # Every window-start state in the dataset (ds.indices: (path_idx, start, end),
    # ds.seq_obs: (n_paths, L, D) GaussianNormalizer-NORMALIZED — same convention
    # scripts/check_df_ckpt.py uses to pull real windows).
    seq_obs = np.asarray(ds.seq_obs)
    idx = np.asarray([(i[0], i[1]) for i in ds.indices], dtype=np.int64)
    states_norm = seq_obs[idx[:, 0], idx[:, 1]]                # (N, D) normalized

    print(f"[{args.env} @ {args.tag}] scanning {states_norm.shape[0]} dataset "
          f"window-start states for grounded solved-count...")
    start_counts = _chunked_count(checker, states_norm, device)
    _print_dist("start-state solved-count distribution", start_counts)

    # Select --n states with count == --min-solved.
    cand_idx = np.where(start_counts == float(args.min_solved))[0]
    if cand_idx.size == 0:
        sys.exit(f"no dataset states have grounded solved-count == "
                 f"{args.min_solved} — cannot run the diagnostic (see the "
                 f"distribution above)")
    rng = np.random.default_rng(args.seed)
    if cand_idx.size < args.n:
        print(f"  only {cand_idx.size} states have count=={args.min_solved} "
              f"(< --n {args.n}); using all of them")
        sel = cand_idx
    else:
        sel = rng.choice(cand_idx, size=args.n, replace=False)
    n_sel = int(sel.size)
    sel_states = states_norm[sel]                              # (n_sel, D) normalized

    # For each state-chunk: x_hist zeros with row0 = state (the MCSS conditioning
    # pattern check_df_ckpt.py uses), repeated --k times per state, hist_len=1
    # (row 0 is the only clean history token) -- mirrors mcts_loop.mcss_waypoints'
    # repeat_interleave for drawing k samples per conditioning state.
    target = args.min_solved + 1
    any_reach = np.zeros(n_sel, dtype=bool)
    max_per_state = np.zeros(n_sel, dtype=np.float64)
    pooled_chunks = []
    n_reach_windows = 0
    n_same_or_better = 0
    for i in range(0, n_sel, STATE_CHUNK):
        chunk = sel_states[i:i + STATE_CHUNK]
        cs = chunk.shape[0]
        s = torch.as_tensor(chunk, dtype=torch.float32, device=device)
        x_hist = torch.zeros((cs * args.k, H, D), dtype=torch.float32, device=device)
        x_hist[:, 0, :] = s.repeat_interleave(args.k, dim=0)
        hist_len = torch.ones(cs * args.k, dtype=torch.long, device=device)
        with torch.no_grad():
            gen = planner.sample(x_hist, hist_len, H, guide=cg_critic, w_cg=args.cg_w)
            wcounts = checker.count(gen).cpu().numpy()          # (cs*k,)
        wcounts = wcounts.reshape(cs, args.k)
        pooled_chunks.append(wcounts.reshape(-1))
        max_per_state[i:i + cs] = wcounts.max(axis=1)
        any_reach[i:i + cs] = (wcounts >= target).any(axis=1)
        n_reach_windows += int((wcounts >= target).sum())
        n_same_or_better += int((wcounts >= args.min_solved).sum())

    pooled = np.concatenate(pooled_chunks)
    print(f"\n[{args.env} @ {args.tag}] conditioned on {n_sel} states with "
          f"grounded count=={args.min_solved}, {args.k} samples each "
          f"({pooled.size} windows total), cg_w={args.cg_w}")
    _print_dist("sampled-window solved-count distribution (pooled)", pooled)
    _print_dist("per-state MAX solved-count distribution", max_per_state)

    n_states_reach = int(any_reach.sum())
    frac_states_reach = n_states_reach / n_sel
    frac_windows_reach = n_reach_windows / pooled.size
    frac_same_or_better = n_same_or_better / pooled.size
    print("\nHEADLINE")
    print(f"  states reaching {target} (ANY of k windows): "
          f"{n_states_reach}/{n_sel} ({frac_states_reach*100:.1f}%)")
    print(f"  windows reaching {target}: {n_reach_windows}/{pooled.size} "
          f"({frac_windows_reach*100:.2f}%)")
    print(f"  windows >= {args.min_solved} (preserves current progress, a "
          f"sanity floor): {n_same_or_better}/{pooled.size} "
          f"({frac_same_or_better*100:.1f}%)")

    print("\nREAD: >0 states reach min_solved+1 => the planner can imagine the "
          "undemonstrated completion — the grounded tree/MCSS arms are worth "
          "closed-loop GPU (run_mcts_compare --value-mode grounded / "
          "--grounded-mcss 1). Zero across the board => the generative model "
          "is also capped by its data; record the boundary and stop.")


if __name__ == "__main__":
    main()
