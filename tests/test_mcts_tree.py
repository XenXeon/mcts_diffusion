"""tests/test_mcts_tree.py

Unit and integration tests for mcts.node and mcts.tree.

Unit tests (46)
---------------
CPU only.  A FakeExpansion class replaces PlannerExpansion — no checkpoint,
no d4rl, no GPU required.  Tests are grouped:
    - TreeConfig validation
    - MCTSNode construction and properties
    - UCB formula
    - MCTSEdge storage by mode
    - Backpropagation
    - Selection
    - Run loop metrics and invariants
    - All three modes produce identical node counts

Integration tests (2)
---------------------
Marked @pytest.mark.integration + @requires_checkpoint.
Load the real maze2d-umaze-v1 checkpoint and run a short tree search.
Skipped automatically when the checkpoint directory is absent.

Run only unit tests:
    pytest tests/test_mcts_tree.py -m "not integration" -v

Run everything (requires checkpoint + GPU):
    pytest tests/test_mcts_tree.py -v
"""
from __future__ import annotations

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcts.expansion import ExpansionResult
from mcts.node import MCTSEdge, MCTSNode, TreeConfig
from mcts.tree import MCTSTree, StepRecord

# ── FakeExpansion ──────────────────────────────────────────────────────────────

class FakeExpansion:
    """Deterministic fake expansion for unit testing.

    Each call returns K trajectories.  Scores are linearly spaced 0.9 → 0.1
    (already sorted descending, mirroring PlannerExpansion).  The next-state
    at position jump_step is distinct across calls so the tree accumulates
    genuinely different nodes.
    """

    def __init__(self, K: int = 3, H: int = 4, obs_dim: int = 4,
                 jump_step: int = 1) -> None:
        self.K = K
        self.H = H
        self.obs_dim = obs_dim
        self.jump_step = jump_step
        self._call_count = 0

    def expand(self, s_norm: torch.Tensor) -> ExpansionResult:
        trajs = torch.zeros(self.K, self.H, self.obs_dim)
        # position-0 = s_norm (mimics fix_mask)
        trajs[:, 0, :] = s_norm.unsqueeze(0).expand(self.K, -1)
        # position jump_step is distinct per call and per candidate
        trajs[:, self.jump_step, 0] = (
            torch.arange(self.K, dtype=torch.float32) * 0.01
            + self._call_count * 0.1
        )
        scores = torch.linspace(0.9, 0.1, self.K)
        self._call_count += 1
        return ExpansionResult(trajs=trajs, scores=scores)

    def expand_batch(self, states: torch.Tensor) -> list:
        return [self.expand(states[i]) for i in range(states.shape[0])]


# ── Shared small config ────────────────────────────────────────────────────────

def make_cfg(mode: str = "state_only", K: int = 3, max_exp: int = 5,
             leaf_batch_size: int = 1,
             ucb_tie_breaking: str = "random") -> TreeConfig:
    return TreeConfig(
        obs_dim=4, horizon=4, child_state_index=1,
        K=K, ucb_c=math.sqrt(2),
        storage_mode=mode, max_expansions=max_exp, device="cpu",
        leaf_batch_size=leaf_batch_size,
        ucb_tie_breaking=ucb_tie_breaking,
    )


def make_fake(K: int = 3) -> FakeExpansion:
    return FakeExpansion(K=K, H=4, obs_dim=4, jump_step=1)


# ── TreeConfig validation ──────────────────────────────────────────────────────

def test_treeconfig_valid():
    cfg = make_cfg("state_only")
    assert cfg.storage_mode == "state_only"


def test_treeconfig_frozen():
    cfg = make_cfg()
    with pytest.raises(Exception):
        cfg.storage_mode = "trajectory_node"  # type: ignore[misc]


def test_treeconfig_invalid_mode():
    with pytest.raises(ValueError, match="storage_mode"):
        TreeConfig(
            obs_dim=4, horizon=4, child_state_index=1,
            K=3, ucb_c=1.0, storage_mode="bad_mode",
            max_expansions=5, device="cpu",
        )


def test_treeconfig_child_state_index_too_large():
    with pytest.raises(ValueError, match="child_state_index"):
        TreeConfig(
            obs_dim=4, horizon=4, child_state_index=4,   # must be < horizon=4
            K=3, ucb_c=1.0, storage_mode="state_only",
            max_expansions=5, device="cpu",
        )


def test_treeconfig_child_state_index_zero():
    with pytest.raises(ValueError, match="child_state_index"):
        TreeConfig(
            obs_dim=4, horizon=4, child_state_index=0,   # 0 → child == parent (self-loop)
            K=3, ucb_c=1.0, storage_mode="state_only",
            max_expansions=5, device="cpu",
        )


def test_treeconfig_child_state_index_negative():
    with pytest.raises(ValueError, match="child_state_index"):
        TreeConfig(
            obs_dim=4, horizon=4, child_state_index=-1,  # negative → silent reverse indexing
            K=3, ucb_c=1.0, storage_mode="state_only",
            max_expansions=5, device="cpu",
        )


# ── MCTSNode construction ──────────────────────────────────────────────────────

def test_node_default_fields():
    cfg = make_cfg()
    s = torch.zeros(4)
    node = MCTSNode(s_norm=s, config=cfg)
    assert node.visit_count == 0
    assert node.value_sum == 0.0
    assert node.is_leaf
    assert node.is_root
    assert node.parent_edge is None


def test_node_s_norm_stored_on_cpu():
    cfg = make_cfg()
    s = torch.ones(4)
    node = MCTSNode(s_norm=s, config=cfg)
    assert node.s_norm.device.type == "cpu"
    assert torch.equal(node.s_norm, s)


def test_node_value_unvisited_is_zero():
    cfg = make_cfg()
    node = MCTSNode(s_norm=torch.zeros(4), config=cfg)
    assert node.value() == 0.0


def test_node_value_after_updates():
    cfg = make_cfg()
    node = MCTSNode(s_norm=torch.zeros(4), config=cfg)
    node.visit_count = 4
    node.value_sum = 2.0
    assert abs(node.value() - 0.5) < 1e-9


# ── UCB formula ───────────────────────────────────────────────────────────────

def test_ucb_unvisited_is_inf():
    cfg = make_cfg()
    parent = MCTSNode(s_norm=torch.zeros(4), config=cfg)
    child = MCTSNode(s_norm=torch.ones(4), config=cfg)
    edge = MCTSEdge(parent=parent, child=child)
    child.parent_edge = edge
    parent.add_child(child, edge)
    parent.visit_count = 5
    # child.visit_count == 0 → UCB = inf
    assert child.ucb(1.0) == float("inf")


def test_ucb_formula_correct():
    cfg = make_cfg()
    parent = MCTSNode(s_norm=torch.zeros(4), config=cfg)
    child = MCTSNode(s_norm=torch.ones(4), config=cfg)
    edge = MCTSEdge(parent=parent, child=child)
    child.parent_edge = edge
    parent.add_child(child, edge)

    parent.visit_count = 10
    child.visit_count = 2
    child.value_sum = 1.6   # value = 0.8

    c = 1.41
    expected = 0.8 + c * math.sqrt(math.log(10) / 2)
    assert abs(child.ucb(c) - expected) < 1e-9


def test_best_child_picks_highest_ucb():
    cfg = make_cfg()
    parent = MCTSNode(s_norm=torch.zeros(4), config=cfg)
    parent.visit_count = 10

    for i, (vc, vs) in enumerate([(3, 2.7), (3, 0.3), (3, 1.5)]):
        child = MCTSNode(s_norm=torch.tensor([float(i), 0, 0, 0]), config=cfg)
        edge = MCTSEdge(parent=parent, child=child)
        child.parent_edge = edge
        child.visit_count = vc
        child.value_sum = vs
        parent.add_child(child, edge)

    # child[0]: value=0.9, child[1]: value=0.1, child[2]: value=0.5
    # All same visit_count → UCB rank = value rank → best is child[0]
    best = parent.best_child(1.41)
    assert abs(best.value() - 0.9) < 1e-9


def test_best_child_random_among_all_unvisited():
    """When all children are unvisited (UCB=inf), best_child picks uniformly at random.

    Previously max() always returned children[0] (greedy-first).  With random
    tie-breaking, different children should be selected across repeated calls.
    """
    torch.manual_seed(0)
    cfg = make_cfg(K=3)
    parent = MCTSNode(s_norm=torch.zeros(4), config=cfg)
    parent.visit_count = 10
    for i in range(3):
        child = MCTSNode(s_norm=torch.tensor([float(i + 1), 0.0, 0.0, 0.0]), config=cfg)
        edge = MCTSEdge(parent=parent, child=child)
        child.parent_edge = edge
        parent.add_child(child, edge)

    # Over 20 draws the set of selected children should contain more than one distinct node.
    # P(all 20 picks are the same child) = (1/3)^19 ≈ 10^-9, negligible.
    selections = set(id(parent.best_child(math.sqrt(2))) for _ in range(20))
    assert len(selections) > 1, "tie-breaking is not random — always returns the same child"


def test_best_child_deterministic_when_clear_winner():
    """When one child has a strictly higher UCB score it is always selected (no randomness)."""
    cfg = make_cfg()
    parent = MCTSNode(s_norm=torch.zeros(4), config=cfg)
    parent.visit_count = 10

    children = []
    for i, (vc, vs) in enumerate([(3, 2.7), (3, 0.3), (3, 1.5)]):
        child = MCTSNode(s_norm=torch.tensor([float(i), 0.0, 0.0, 0.0]), config=cfg)
        edge = MCTSEdge(parent=parent, child=child)
        child.parent_edge = edge
        child.visit_count = vc
        child.value_sum = vs
        parent.add_child(child, edge)
        children.append(child)

    # child[0] value=0.9 is the unique maximum — must always win
    for _ in range(10):
        assert parent.best_child(math.sqrt(2)) is children[0]


def test_best_child_greedy_always_returns_first():
    """With ucb_tie_breaking='greedy', ties always resolve to the first child (highest critic score)."""
    cfg = make_cfg(K=3, ucb_tie_breaking="greedy")
    parent = MCTSNode(s_norm=torch.zeros(4), config=cfg)
    parent.visit_count = 10
    children = []
    for i in range(3):
        child = MCTSNode(s_norm=torch.tensor([float(i + 1), 0.0, 0.0, 0.0]), config=cfg)
        edge = MCTSEdge(parent=parent, child=child)
        child.parent_edge = edge
        parent.add_child(child, edge)
        children.append(child)

    # All unvisited → UCB=inf for all → tie → greedy picks children[0] every time
    for _ in range(10):
        assert parent.best_child(math.sqrt(2)) is children[0]


def test_treeconfig_invalid_tie_breaking():
    with pytest.raises(ValueError, match="ucb_tie_breaking"):
        TreeConfig(
            obs_dim=4, horizon=4, child_state_index=1,
            K=3, ucb_c=1.0, storage_mode="state_only",
            max_expansions=5, device="cpu",
            ucb_tie_breaking="invalid",
        )


def test_first_selection_after_expansion_picks_any_child():
    """After expanding root, _select returns one of root's children (random, not fixed to children[0])."""
    torch.manual_seed(0)
    cfg = make_cfg(K=3)
    fake = make_fake(K=3)
    tree = MCTSTree(torch.zeros(4), fake, cfg)
    tree._expand(tree.root)
    selected = tree._select()
    assert selected in tree.root.children


# ── Storage modes ─────────────────────────────────────────────────────────────

def test_mode_a_no_traj_on_node():
    cfg = make_cfg("state_only")
    fake = make_fake()
    tree = MCTSTree(torch.zeros(4), fake, cfg)
    tree._expand(tree.root)
    for child in tree.root.children:
        assert child.traj is None


def test_mode_b_traj_on_node():
    cfg = make_cfg("trajectory_node")
    fake = make_fake()
    tree = MCTSTree(torch.zeros(4), fake, cfg)
    tree._expand(tree.root)
    for child in tree.root.children:
        assert child.traj is not None
        assert child.traj.shape == (cfg.horizon, cfg.obs_dim)


def test_mode_c_no_traj_on_node():
    cfg = make_cfg("state_edge_trajectory")
    fake = make_fake()
    tree = MCTSTree(torch.zeros(4), fake, cfg)
    tree._expand(tree.root)
    for child in tree.root.children:
        assert child.traj is None


def test_mode_c_traj_on_edge():
    cfg = make_cfg("state_edge_trajectory")
    fake = make_fake()
    tree = MCTSTree(torch.zeros(4), fake, cfg)
    tree._expand(tree.root)
    for edge in tree.root.edges:
        assert edge.traj is not None
        assert edge.traj.shape == (cfg.horizon, cfg.obs_dim)
        assert edge.score is not None


def test_mode_a_no_traj_on_edge():
    cfg = make_cfg("state_only")
    fake = make_fake()
    tree = MCTSTree(torch.zeros(4), fake, cfg)
    tree._expand(tree.root)
    for edge in tree.root.edges:
        assert edge.traj is None
        assert edge.score is None


def test_child_parent_edge_links_back_to_parent():
    cfg = make_cfg()
    fake = make_fake()
    tree = MCTSTree(torch.zeros(4), fake, cfg)
    tree._expand(tree.root)
    for child in tree.root.children:
        assert child.parent_edge is not None
        assert child.parent_edge.parent is tree.root
        assert child.parent_edge.child is child


# ── Backpropagation ───────────────────────────────────────────────────────────

def test_backprop_increments_visit_count():
    cfg = make_cfg()
    fake = make_fake()
    tree = MCTSTree(torch.zeros(4), fake, cfg)
    tree._expand(tree.root)
    child = tree.root.children[0]
    tree._backprop(child, 0.7)
    assert child.visit_count == 1
    assert tree.root.visit_count == 1


def test_backprop_accumulates_value():
    cfg = make_cfg()
    fake = make_fake()
    tree = MCTSTree(torch.zeros(4), fake, cfg)
    tree._expand(tree.root)
    child = tree.root.children[0]
    tree._backprop(child, 0.4)
    tree._backprop(child, 0.6)
    assert abs(child.value() - 0.5) < 1e-9


def test_backprop_from_depth2_reaches_root():
    cfg = make_cfg()
    fake = make_fake()
    tree = MCTSTree(torch.zeros(4), fake, cfg)
    tree._expand(tree.root)
    grandchild_leaf = tree.root.children[0]
    tree._expand(grandchild_leaf)
    ggchild = grandchild_leaf.children[0]
    tree._backprop(ggchild, 0.5)
    # root, grandchild_leaf, ggchild all updated
    assert ggchild.visit_count == 1
    assert grandchild_leaf.visit_count == 1
    assert tree.root.visit_count == 1


# ── Selection ─────────────────────────────────────────────────────────────────

def test_select_returns_root_when_leaf():
    cfg = make_cfg()
    fake = make_fake()
    tree = MCTSTree(torch.zeros(4), fake, cfg)
    assert tree._select() is tree.root


def test_select_after_expansion_returns_unvisited_child():
    cfg = make_cfg()
    fake = make_fake()
    tree = MCTSTree(torch.zeros(4), fake, cfg)
    tree._expand(tree.root)
    tree._backprop(tree.root.children[0], 0.5)   # mark one child visited

    selected = tree._select()
    # root is no longer a leaf — selected must be one of root's children
    assert selected in tree.root.children


def test_selected_depth_root_is_zero():
    cfg = make_cfg()
    fake = make_fake()
    tree = MCTSTree(torch.zeros(4), fake, cfg)
    assert tree._node_depth(tree.root) == 0


def test_selected_depth_child_is_one():
    cfg = make_cfg()
    fake = make_fake()
    tree = MCTSTree(torch.zeros(4), fake, cfg)
    tree._expand(tree.root)
    for child in tree.root.children:
        assert tree._node_depth(child) == 1


# ── Run loop ──────────────────────────────────────────────────────────────────

def test_run_returns_correct_n_records():
    cfg = make_cfg(max_exp=5)
    records = MCTSTree(torch.zeros(4), make_fake(), cfg).run()
    assert len(records) == 5


def test_n_nodes_after_step_zero():
    cfg = make_cfg(K=3, max_exp=1)
    records = MCTSTree(torch.zeros(4), make_fake(K=3), cfg).run()
    # root + K children
    assert records[0].n_nodes == 1 + 3


def test_n_nodes_grows_by_K_each_step():
    K = 3
    cfg = make_cfg(K=K, max_exp=5)
    records = MCTSTree(torch.zeros(4), make_fake(K=K), cfg).run()
    for i, rec in enumerate(records):
        assert rec.n_nodes == 1 + (i + 1) * K, (
            f"step {i}: expected {1 + (i+1)*K} nodes, got {rec.n_nodes}"
        )


def test_cumulative_best_nondecreasing():
    cfg = make_cfg(max_exp=5)
    records = MCTSTree(torch.zeros(4), make_fake(), cfg).run()
    for i in range(len(records) - 1):
        assert records[i].cumulative_best <= records[i + 1].cumulative_best


def test_step_record_fields_nonnegative():
    cfg = make_cfg(max_exp=3)
    records = MCTSTree(torch.zeros(4), make_fake(), cfg).run()
    for rec in records:
        assert rec.n_nodes > 0
        assert rec.tree_depth >= 0
        assert rec.expand_time >= 0.0
        assert rec.wall_time >= 0.0


def test_tree_depth_reaches_two():
    # With K=3, step 0 expands root → depth 1.
    # Step 1: _select picks root's first unvisited child (leaf at depth 1),
    # expands it → depth becomes 2.
    cfg = make_cfg(K=3, max_exp=2)
    records = MCTSTree(torch.zeros(4), make_fake(K=3), cfg).run()
    assert records[-1].tree_depth >= 2


def test_all_modes_same_n_nodes():
    root = torch.zeros(4)
    for mode in ("state_only", "trajectory_node", "state_edge_trajectory"):
        cfg = make_cfg(mode, K=3, max_exp=4)
        records = MCTSTree(root, make_fake(K=3), cfg).run()
        assert records[-1].n_nodes == 1 + 4 * 3, (
            f"mode={mode}: expected {1+4*3}, got {records[-1].n_nodes}"
        )


def test_theoretical_floats_mode_a_is_zero():
    cfg = make_cfg("state_only")
    assert MCTSTree.theoretical_floats(cfg, n_nodes=100) == 0


def test_theoretical_floats_mode_b_correct():
    cfg = make_cfg("trajectory_node")
    n = 100
    expected = n * cfg.horizon * cfg.obs_dim
    assert MCTSTree.theoretical_floats(cfg, n_nodes=n) == expected


def test_theoretical_floats_mode_c_correct():
    cfg = make_cfg("state_edge_trajectory")
    n = 100
    expected = (n - 1) * cfg.horizon * cfg.obs_dim
    assert MCTSTree.theoretical_floats(cfg, n_nodes=n) == expected


# ── Verified claims — document observed behaviour ─────────────────────────────
#
# These tests pin the three properties identified in the Claude cross-review:
#   C1: each run() step makes exactly one sequential expand() call (no batching)
#   C2: UCB ties (all unvisited → UCB=inf) resolved by insertion order → greedy-first
#   C3: n_unique_states == n_nodes in practice (continuous states never collide)

def test_each_run_step_makes_exactly_one_expand_call():
    """Each step in run() calls expansion.expand() exactly once — no batching.

    With max_expansions=N there are N sequential GPU calls.  Leaf parallelisation
    (batching multiple leaves into one planner.sample() call) would reduce this
    to N/batch_size calls.
    """
    class CountingFake(FakeExpansion):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.call_count = 0

        def expand(self, s_norm):
            self.call_count += 1
            return super().expand(s_norm)

    max_exp = 5
    cfg = make_cfg(K=3, max_exp=max_exp)
    fake = CountingFake(K=3, H=4, obs_dim=4)
    MCTSTree(torch.zeros(4), fake, cfg).run()
    assert fake.call_count == max_exp


def test_expand_batch_returns_n_results_with_correct_shapes():
    """FakeExpansion.expand_batch returns one ExpansionResult per input state."""
    fake = make_fake(K=3)
    N = 4
    states = torch.rand(N, 4)
    results = fake.expand_batch(states)
    assert len(results) == N
    for r in results:
        assert r.trajs.shape == (3, 4, 4)   # (K, H, obs_dim)
        assert r.scores.shape == (3,)
        assert r.scores[0] >= r.scores[-1]  # sorted descending


def test_batched_run_correct_n_nodes():
    """With leaf_batch_size=2, n_nodes still grows by K per expansion step."""
    K, max_exp, B = 3, 6, 2
    cfg = make_cfg(K=K, max_exp=max_exp, leaf_batch_size=B)
    records = MCTSTree(torch.zeros(4), make_fake(K=K), cfg).run()
    assert len(records) == max_exp
    for i, rec in enumerate(records):
        assert rec.n_nodes == 1 + (i + 1) * K


def test_batched_run_no_duplicate_expansion():
    """When batch_size > available leaves, duplicates are skipped — each leaf expanded once."""
    # batch_size=5 > K=3 leaves at step 0 (only root exists).
    # Without deduplication root would get K children added batch_size times.
    K, B = 3, 5
    cfg = make_cfg(K=K, max_exp=1, leaf_batch_size=B)
    tree = MCTSTree(torch.zeros(4), make_fake(K=K), cfg)
    tree.run()
    # root should have exactly K children, not K*B
    assert len(tree.root.children) == K


def test_batched_run_fewer_gpu_calls():
    """With leaf_batch_size=B, run() makes max_expansions/B expand_batch calls, not max_expansions."""
    class BatchCountingFake(FakeExpansion):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.batch_call_count = 0

        def expand_batch(self, states):
            self.batch_call_count += 1
            return super().expand_batch(states)

    max_exp, B = 6, 2
    cfg = make_cfg(K=3, max_exp=max_exp, leaf_batch_size=B)
    fake = BatchCountingFake(K=3, H=4, obs_dim=4)
    MCTSTree(torch.zeros(4), fake, cfg).run()
    # Batching always reduces GPU calls vs sequential (max_exp calls).
    # Exact count varies because the first batch degenerates to N=1 (only root exists),
    # but subsequent batches are full-size once the tree has enough leaves.
    assert fake.batch_call_count < max_exp





# ── Integration tests — real checkpoint ───────────────────────────────────────

CKPT_DIR = os.path.join(
    os.path.dirname(__file__), "..",
    "results",
    "veteran_d4rl_maze2d_H32_Jump15_next1_MCSS_transformer_d2_width256_separate_dpTrue",
    "maze2d-umaze-v1",
)

integration = pytest.mark.integration
requires_checkpoint = pytest.mark.skipif(
    not os.path.isdir(CKPT_DIR),
    reason=f"Checkpoint directory not found: {CKPT_DIR}",
)


@pytest.fixture(scope="module")
def real_expansion_for_tree():
    """PlannerExpansion backed by the real maze2d-umaze-v1 checkpoint (CPU)."""
    import torch
    from cleandiffuser.diffusion import ContinuousDiffusionSDE
    from cleandiffuser.nn_diffusion import DiT1d
    from cleandiffuser.utils import DVHorizonCritic
    from mcts.expansion import ExpansionConfig, PlannerExpansion

    cfg = ExpansionConfig(
        K=50, horizon=32, obs_dim=4, planner_dim=4,
        solver="ddim", sample_steps=20, temperature=1.0,
        use_ema=True, device="cpu",
    )
    nn_diff = DiT1d(
        cfg.planner_dim, emb_dim=128, d_model=256,
        n_heads=4, depth=2, timestep_emb_type="fourier",
    )
    fix_mask = torch.zeros((cfg.horizon, cfg.planner_dim))
    fix_mask[0, : cfg.obs_dim] = 1.0
    planner = ContinuousDiffusionSDE(
        nn_diff, nn_condition=None, fix_mask=fix_mask,
        loss_weight=torch.ones((cfg.horizon, cfg.planner_dim)),
        ema_rate=0.9999, device="cpu",
        predict_noise=True, noise_schedule="linear",
    )
    planner.load(os.path.join(CKPT_DIR, "planner_ckpt_1000000.pt"))
    planner.eval()

    critic_ckpt = torch.load(
        os.path.join(CKPT_DIR, "critic_ckpt_1000000.pt"), map_location="cpu"
    )
    from cleandiffuser.utils import DVHorizonCritic
    critic = DVHorizonCritic(
        cfg.planner_dim, emb_dim=128, d_model=256,
        n_heads=4, depth=2, norm_type="pre",
    ).to("cpu")
    critic.load_state_dict(critic_ckpt["critic"])
    critic.eval()

    return PlannerExpansion(planner, critic, cfg)


@integration
@requires_checkpoint
def test_integration_tree_one_step(real_expansion_for_tree):
    """One expansion on the real model: node count, depth, and score range."""
    torch.manual_seed(0)
    cfg = TreeConfig(
        obs_dim=4, horizon=32, child_state_index=1, K=50,
        ucb_c=math.sqrt(2), storage_mode="state_only",
        max_expansions=1, device="cpu",
    )
    root_s = torch.zeros(4)
    tree = MCTSTree(root_s, real_expansion_for_tree, cfg)
    records = tree.run()

    assert len(records) == 1
    rec = records[0]
    assert rec.n_nodes == 51           # root + 50 children
    assert rec.tree_depth == 1
    assert -1.5 <= rec.leaf_best_score <= 1.5
    assert -1.5 <= rec.leaf_mean_score <= 1.5
    assert rec.cumulative_best == rec.leaf_best_score


@integration
@requires_checkpoint
def test_integration_all_modes_same_n_nodes(real_expansion_for_tree):
    """All three storage modes produce identical n_nodes for 2 expansions."""
    root_s = torch.zeros(4)
    n_nodes_per_mode = {}
    for mode in ("state_only", "trajectory_node", "state_edge_trajectory"):
        torch.manual_seed(42)
        cfg = TreeConfig(
            obs_dim=4, horizon=32, child_state_index=1, K=50,
            ucb_c=math.sqrt(2), storage_mode=mode,
            max_expansions=2, device="cpu",
        )
        records = MCTSTree(root_s, real_expansion_for_tree, cfg).run()
        n_nodes_per_mode[mode] = records[-1].n_nodes

    assert len(set(n_nodes_per_mode.values())) == 1, (
        f"n_nodes differs across modes: {n_nodes_per_mode}"
    )
