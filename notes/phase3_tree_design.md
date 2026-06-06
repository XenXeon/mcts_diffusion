# Phase 3 Tree Design — MCTS over DV Planner

**Date:** 2026-06-05

---

## What Phase 3 Builds

An MCTS tree that uses the Phase 2 expansion primitive (`mcts/expansion.py`) as its leaf evaluator. The tree is structured as an ablation: a single `storage_mode` flag selects one of three designs, all sharing the same UCB selection, backpropagation, and run loop.

---

## Storage Mode Taxonomy

| Mode | Config string | Node stores | Edge stores | Traj floats / node (H=32, D=4) |
|------|---------------|-------------|-------------|-------------------------------|
| A | `state_only` | s_norm only | — | 0 |
| B | `trajectory_node` | s_norm + traj (H, D) | — | 128 |
| C | `state_edge_trajectory` | s_norm only | traj (H, D) + score | 128 (on edge; n_edges = n_nodes − 1) |

Mode B duplicates what A has plus the trajectory on each node.
Mode C mirrors B's data but moves it to the edge, which is the natural ownership (a trajectory belongs to the transition, not the destination state).

---

## UCB1 Formula

```
UCB(child) = value(child) + c * sqrt( ln(N_parent) / N_child )
```

- `value(child) = value_sum / visit_count` (incremental mean of backpropagated scores)
- Unvisited nodes (`visit_count == 0`) return `+inf` — always expanded before re-visiting
- `c = sqrt(2)` (standard UCB1 constant)

---

## Backpropagation Value Choice

The expansion primitive generates K=50 trajectories per call, each scored by the critic. Two options:

- **Best score** — consistent with MCSS (Phase 1 used argmax). Overestimates leaf quality.
- **Mean score** — Monte Carlo estimate of the leaf's expected value under the planner distribution. Used here.

Mean is the correct choice for MCTS value estimation: it treats the K candidates as iid samples from the planner and estimates the expected critic return under the current policy.

---

## Child State Extraction

The planner generates trajectories of H=32 waypoints, each separated by M=15 dense env steps (the dataset `stride`). The mapping is:

```
traj[0]  = current state            (0 dense steps)
traj[1]  = next planning target     (15 dense steps ahead)  ← child state
traj[2]  = two jumps ahead          (30 dense steps)
...
traj[k]  = k planning jumps ahead   (k × 15 dense steps)
```

The production pipeline confirms this: `veteran_d4rl_maze2d.py` lines 418/438 use `traj[:, 1, :]` as the immediate next target state for the inverse-dynamics policy.

For the MCTS tree, each edge represents **one planning step** (the agent executes M=15 dense steps to reach the next replanning state). So:

```python
next_s = traj[child_state_index, :obs_dim]   # child_state_index = 1
```

**Do not confuse** `child_state_index=1` with `M=15`. M=15 is the dense-step stride between waypoints — a dataset constant. `child_state_index=1` is the trajectory array index for the next waypoint. Using index 15 would jump 15×15=225 dense steps per tree edge, skipping 15 replanning opportunities.

## UCB Tie-Breaking Behaviour

When a node is first expanded, all K children start with `visit_count=0`, giving UCB=+inf. Python's `max()` on equal values returns the first maximum found, which is `children[0]` — the highest-scoring candidate (since `PlannerExpansion` returns trajectories sorted descending by critic score).

This means early traversal is **score-ordered**: the best-critic-scored child is always visited before re-visiting explored subtrees. This is not a bug — it is greedy-first exploration — but it means the tree initially behaves more like beam search than balanced MCTS. UCB only diversifies once all children have been visited at least once (after K expansions from a given parent). This behaviour is worth noting in ablation analysis when comparing depth vs. breadth growth rates.

---

## Theoretical Memory (trajectory data only, fp32)

For a tree with `n` nodes (`n_edges = n − 1`):

| Mode | Formula | n=3001 (60 exp × K=50) |
|------|---------|----------------------|
| A | `0` | 0 floats |
| B | `n × H × D` | 384,128 floats ≈ 1.5 MB |
| C | `(n−1) × H × D` | 384,000 floats ≈ 1.5 MB |

Modes B and C are near-identical in memory. The difference is semantic (node vs. edge ownership) and access pattern (retrieving a trajectory for a node vs. for an edge).

---

## Known Limitation: Critic Score Saturation

Phase 1 confirmed that the critic scores generated plans in `[0.50, 0.97]` — all positive, all high. In a tree where every leaf scores ≥ 0.5, UCB's exploration term must dominate to avoid the tree collapsing to a single deep branch. With `c = sqrt(2)` and scores in `[0, 1]`, exploration is meaningful but may need tuning (Phase 4/5 concern).

---

## Files Created in Phase 3

| File | Purpose |
|------|---------|
| `mcts/node.py` | `MCTSNode`, `MCTSEdge`, `TreeConfig` |
| `mcts/tree.py` | `MCTSTree` (selection, expand, backprop, run), `StepRecord` |
| `mcts/__init__.py` | Updated exports |
| `tests/test_mcts_tree.py` | 26 unit + 2 integration tests |
| `scripts/phase3_ablation.py` | Ablation runner; writes `results/phase3/ablation_{mode}.csv` |
| `notes/phase3_tree_design.md` | This file |
| `Makefile` | Added `test-mcts-tree-unit`, `test-mcts-tree`, `ablation-phase3` |
