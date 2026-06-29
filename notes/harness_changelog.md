# Harness changelog — the code behind the OLD cells vs the NEW (scale-up) cells

Purpose: record EXACTLY what changed in the evaluation harness between the original
Phase-E cells (`results/{mcss,mcts}_antmaze_*.json`, incl. the b16 = 96% cell) and the
scale-up cells (`results/scale_*.json`), so old-vs-new differences are never mistaken
for an algorithm change. (Requested as the "no_harness" reference file.)

## TL;DR

**The sampler, the search, and the models were not touched.** `Sampler.mcss_waypoints`,
`Sampler.mcts_waypoints`, `Sampler.expand_fn`, `Sampler.policy_action`,
`mcts/value_forest.py` (the whole MCTS engine), `mcts/value_net.py`, and all four
checkpoints (planner/critic/policy 1M, state-value latest) are byte-identical between
the two sets of runs. Every old budget cell (b4=60, b8=80, b16=96) and every old MCSS
cell (k50=76, k144=80, k272=84) ran on the SAME old harness, so the old comparison was
internally consistent.

Exactly ONE change affects what the environment does:

> `gym.vector.make(env_name, n_envs)`  →  `gym.vector.make(env_name, n_envs, asynchronous=False)`

The async vector env constructs sub-envs in worker processes, so the goal-jitter draws
(antmaze samples its goal from the global `np.random` at construction —
`d4rl/locomotion/maze_env.py:set_target_goal`) come from worker RNG state. The sync env
constructs sub-envs sequentially in the main process, consuming one re-seeded global
stream. **Same scenario DISTRIBUTION (goal ≈ eval corner ± jitter), different realized
scenario SETS.** This is why old and new cells with the same `--seed` are not the same
25/50 scenarios, and why the old scenario sets can never be exactly recovered (the
worker RNG states were never recorded).

Every other change is observational or inert:

| # | Change (old → new) | Behavioural effect |
|---|---|---|
| 1 | async → sync vector env | **changes realized scenario draw** (see above); env dynamics identical |
| 2 | `env.seed` failure silently `pass`ed → loud warning + explicit `e.seed(seed+i)`, `e.action_space.seed(seed+i)` per sub-env | none on antmaze (those seed paths are deprecated no-ops in the pinned gym/d4rl; verified empirically — start jitter stays unpaired) |
| 3 | no per-rollout logging → records `success`, `reach_step`, `starts`, `goals` per rollout + `reach_err` | none (attribute reads only, no RNG consumed) |
| 4 | `max_path_length` from family-level SPECS constant → read from env TimeLimit | none for antmaze (both = 1000); fixes maze2d-umaze/medium (300/600, were 800) |
| 5 | local SPECS/TARGET_CFG copies → imported from `mcts/specs.py` | none (identical values) |
| 6 | JSON metadata: added planner/critic/policy/value ckpt steps, ckpt dir, git commit, max_path_length | none |

Plus: n=25 → n=50 per cell, and 1 seed → 3 seeds. The diffusion/torch RNG streams also
differ between any two runs regardless (CUDA nondeterminism — the OLD harness itself
showed ±2 rollouts on identical configs: 76% vs 84% MCSS k50 replicates).

## Why the absolute numbers moved (84/96 → 72.0/83.3 pooled)

Both arms moved DOWN together (−12pp MCSS, −13pp MCTS) while the gap was preserved
(+12pp old, +11.3pp new pooled, exact McNemar p=0.021, n=150 paired). A parallel shift
preserving the gap is the signature of (a) the old single n=25 scenario set being an
easy/lucky draw and (b) ordinary binomial noise at n=25 — not of a harness or
algorithm change. Consistency checks: old-vs-new z = 1.14 (MCSS), 1.66 (MCTS) — both
within sampling noise.

## The old run_episodes, verbatim

The old cells were produced by this exact function (only the parts that later changed
are shown in full; `Sampler` and `load_models` differed only as per the table above):

```python
def run_episodes(sampler: Sampler, method: str, n_envs: int, n_episodes: int,
                 seed: int = 0, max_steps: Optional[int] = None,
                 verbose: bool = True) -> Dict[str, Any]:
    import gym
    from pipelines.utils import set_seed

    assert method in ("mcss", "mcts")
    m = sampler.m
    env_name = m["env_name"]
    normalizer = m["normalizer"]
    env_single = m["env_single"]
    max_t = max_steps or m["max_path_length"]

    set_seed(seed)
    env = gym.vector.make(env_name, n_envs)          # <-- async (the one real difference)
    try:
        env.seed(seed)
    except Exception:
        pass                                          # <-- silent

    all_success: List[np.ndarray] = []
    t0 = time.perf_counter()
    for ep in range(n_episodes):
        obs = env.reset()
        ep_rew = np.zeros(n_envs, dtype=np.float64)
        active = np.ones(n_envs, dtype=bool)   # still in the FIRST episode (count rewards)
        for t in range(max_t):
            s_norm = normalizer.normalize(obs).astype(np.float32)   # (n_envs, obs_dim)
            if method == "mcss":
                next_wp = sampler.mcss_waypoints(s_norm)
            else:
                next_wp = sampler.mcts_waypoints(s_norm)
            act = sampler.policy_action(s_norm, next_wp)
            obs, rew, done, info = env.step(act)
            # gym.vector auto-resets an env on done; count rewards only within each env's
            # first episode (freeze once done) so a reset second episode can't be mixed in.
            ep_rew += np.asarray(rew, dtype=np.float64) * active
            active &= ~np.asarray(done, dtype=bool)
            if not active.any():
                break
        succ = np.clip(ep_rew, 0.0, 1.0)
        all_success.append(succ)
        if verbose:
            print(f"  [{method}] ep {ep+1}/{n_episodes}  reach={succ.mean()*100:5.1f}%  "
                  f"elapsed={time.perf_counter()-t0:6.0f}s")
    env.close()

    flat = np.concatenate(all_success)
    norm = np.array([env_single.get_normalized_score(x) for x in flat]) * 100.0
    out = dict(method=method, n_rollouts=int(flat.size),
               reach_pct=float(flat.mean() * 100.0),
               norm_mean=float(norm.mean()),
               norm_err=float(norm.std() / np.sqrt(flat.size)),
               wall_s=round(time.perf_counter() - t0, 1))     # <-- aggregates only
    if verbose:
        print(f"  [{method}] DONE  reach={out['reach_pct']:.1f}%  "
              f"norm={out['norm_mean']:.1f}±{out['norm_err']:.1f}  "
              f"(n={out['n_rollouts']}, {out['wall_s']:.0f}s)")
    return out
```

To literally re-run the old protocol, paste this over the current `run_episodes` in a
scratch copy (do NOT fork the live harness file — two diverging harnesses is how
old-vs-new confusion starts). Note that even the old code cannot reproduce the old
NUMBERS: the async worker RNG states behind the old scenario draws were never recorded,
and the old harness itself replicated only to ±2 rollouts at n=25.

## What the dissertation should say

The n=25 cells (84/96) and the n=150 paired grid are different scenario samples from
the same distribution measured by the same algorithm; report the paired grid as the
headline (+11.3pp, p=0.021, fixes=33/42=79% of greedy failures, breaks=16/108=15%) and
the old cells as the preliminary experiment that motivated it.
