# Phase 2 Verification — Expansion Primitive

**Date:** 2026-06-05  
**Verdict:** PASS — all 18 tests green, smoke test clean on real checkpoint

---

## What Phase 2 Built

`mcts/expansion.py` — stateless expansion primitive wrapping the DV planner and critic.  
Given a normalised start state `s_norm (obs_dim,)`, returns `K` candidate trajectories ranked by critic score.

Key contracts verified:
- `fix_mask` clamps trajectory position-0 to `s_norm` (max-abs = 0.00 in production run)
- Scores are 1-D `(K,)`, sorted descending, no gradient graph
- Two sequential `expand()` calls produce distinct trajectories (stochastic)
- `ExpansionConfig` is frozen (mutation raises)

---

## Bugs Fixed Before Running

| File | Bug | Fix |
|------|-----|-----|
| `tests/test_mcts_expansion.py` | Critic loaded with `load_state_dict(torch.load(path))` — missing key unwrap | `ckpt = torch.load(path); load_state_dict(ckpt["critic"])` |
| `scripts/phase2_smoke_test.py` | Same critic load bug | Same fix |
| `tests/test_mcts_expansion.py` | `integration = pytest.mark.skipif(...)` — only adds `skipif`, not `integration` marker; `-m "not integration"` did not filter them | Split into `integration = pytest.mark.integration` + `requires_checkpoint = pytest.mark.skipif(...)` |
| `pyproject.toml` | `integration` marker unregistered — pytest warned on unknown mark | Added `[tool.pytest.ini_options]` markers block |
| `scripts/phase2_smoke_test.py` | Missing prerequisite files exited 0 (silent pass) | Changed to exit 1 with `[FAIL]` prefix |

---

## Run 1 — Unit tests (CPU, no checkpoint)

```powershell
docker run --rm `
  -v "${PWD}:/workspace" `
  -w /workspace `
  cleandiffuser:dev `
  pytest tests/test_mcts_expansion.py -m "not integration" -v
```

```
collected 18 items / 4 deselected / 14 selected
14 passed, 4 deselected in 6.54s
```

All 14 unit tests passed. 4 integration tests correctly deselected by marker.

---

## Run 2 — Smoke test (GPU, real checkpoint)

```powershell
docker run --gpus all --rm `
  -v "${PWD}:/workspace" `
  -v "$env:USERPROFILE\.d4rl:/root/.d4rl" `
  -w /workspace `
  cleandiffuser:dev `
  python scripts/phase2_smoke_test.py
```

```
Start state from Phase 1 row 0: traj=1295 offset=20
  s_norm : [-1.0514  -0.0732  -0.0435   2.0082]
  s_raw  : [ 0.9868   2.3225  -0.0960   4.9555]

  Planner + critic loaded on cuda:0.
  Wall time : 0.85s
  trajs shape : (50, 32, 4)
  best_score  : 0.8010
  scores (top-5): [0.8010  0.7847  0.6459  0.6293  0.6263]

  [PASS] fix_mask: max-abs = 0.00e+00 < 1e-4
  [PASS] scores descending
  [PASS] trajs shape: (50, 32, 4)
  [PASS] scores shape: (50,)
  [PASS] planner stochastic: planner_self_l2 = 1.0611

  Phase 1 recorded score_gen : 0.800988
  expand() best_score        : 0.800988

[PASS] All smoke test assertions passed.
```

Score matches Phase 1 exactly (deterministic seed). Wall time 0.85s per expansion = baseline cost per MCTS node.

---

## Run 3 — Full suite (GPU, real checkpoint)

```powershell
docker run --gpus all --rm `
  -v "${PWD}:/workspace" `
  -v "$env:USERPROFILE\.d4rl:/root/.d4rl" `
  -w /workspace `
  cleandiffuser:dev `
  pytest tests/test_mcts_expansion.py -v
```

```
collected 18 items
18 passed in 6.36s
```

All 4 integration tests passed on GPU with real checkpoint.

---

## Phase 3 Entry Conditions

The expansion primitive is verified and ready to be used as a building block. Phase 3 will build:

1. **Tree node structure** — `MCTSNode` holding `s_norm`, visit count, value estimate, parent/children links
2. **UCB selection** — traverse from root to leaf using UCB1 or PUCT
3. **Backpropagation** — update visit counts and value estimates up the tree after each expansion
