"""tests/test_mcts_expansion.py

Unit and integration tests for mcts.expansion.PlannerExpansion.

Unit tests (14)
---------------
Run entirely on CPU with small synthetic models (obs_dim=8, horizon=4, K=3).
No checkpoint loading, no d4rl, no gym.  Fast enough for CI.

Integration tests (4)
---------------------
Marked @pytest.mark.integration + @requires_checkpoint.  Load the real
maze2d-umaze-v1 checkpoint and run one expansion with production config.
Skipped automatically when the checkpoint directory is absent (e.g. on a
fresh clone without model weights).

Run only unit tests:
    pytest tests/test_mcts_expansion.py -m "not integration"

Run everything (requires checkpoint):
    pytest tests/test_mcts_expansion.py
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcts.expansion import ExpansionConfig, ExpansionResult, PlannerExpansion

# ── Shared constants ───────────────────────────────────────────────────────────
DEVICE = "cpu"

# Small synthetic config — enough to exercise all code paths quickly
SMALL_CFG = ExpansionConfig(
    K=3,
    horizon=4,
    obs_dim=8,
    planner_dim=8,
    solver="ddim",
    sample_steps=2,
    temperature=1.0,
    use_ema=False,   # no EMA on randomly-initialised model
    device=DEVICE,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_expansion():
    """PlannerExpansion backed by a small randomly-initialised planner + critic."""
    from cleandiffuser.diffusion import ContinuousDiffusionSDE
    from cleandiffuser.nn_diffusion import DiT1d
    from cleandiffuser.utils import DVHorizonCritic

    cfg = SMALL_CFG

    nn_diff = DiT1d(
        cfg.obs_dim, emb_dim=32, d_model=64, n_heads=2,
        depth=2, timestep_emb_type="fourier",
    )
    fix_mask = torch.zeros((cfg.horizon, cfg.planner_dim))
    fix_mask[0, : cfg.obs_dim] = 1.0
    loss_weight = torch.ones((cfg.horizon, cfg.planner_dim))

    planner = ContinuousDiffusionSDE(
        nn_diff, nn_condition=None,
        fix_mask=fix_mask, loss_weight=loss_weight,
        ema_rate=0.9999, device=DEVICE,
        predict_noise=True, noise_schedule="linear",
    )
    planner.eval()

    critic = DVHorizonCritic(
        cfg.planner_dim, emb_dim=32, d_model=64, n_heads=2, depth=1,
    ).to(DEVICE)
    critic.eval()

    return PlannerExpansion(planner, critic, cfg)


@pytest.fixture(scope="module")
def sample_state():
    """A fixed normalised start state for unit tests."""
    torch.manual_seed(0)
    return torch.randn(SMALL_CFG.obs_dim)


# ── Unit tests — shapes ────────────────────────────────────────────────────────

def test_output_shapes(small_expansion, sample_state):
    """trajs must be (K, H, planner_dim); scores must be 1-D (K,)."""
    cfg = SMALL_CFG
    result = small_expansion.expand(sample_state)
    assert result.trajs.shape == (cfg.K, cfg.horizon, cfg.planner_dim), (
        f"Expected trajs shape ({cfg.K}, {cfg.horizon}, {cfg.planner_dim}), "
        f"got {tuple(result.trajs.shape)}"
    )
    assert result.scores.shape == (cfg.K,), (
        f"Expected scores shape ({cfg.K},), got {tuple(result.scores.shape)}"
    )


def test_scores_are_1d_not_2d(small_expansion, sample_state):
    """critic returns (K,1); squeeze must give (K,) not (K,1)."""
    result = small_expansion.expand(sample_state)
    assert result.scores.dim() == 1


# ── Unit tests — fix_mask contract ────────────────────────────────────────────

def test_fix_mask_clamped(small_expansion, sample_state):
    """Position-0 of every trajectory must equal the start state.

    This is the most critical invariant: if fix_mask is broken, every critic
    score and every L2 metric from Phase 1 onward is computed against the wrong
    start state.  Tolerance matches the Phase 1 mandatory assertion (1e-4).
    """
    result = small_expansion.expand(sample_state)
    s = sample_state.to(DEVICE)
    cfg = SMALL_CFG
    diff = (result.trajs[:, 0, : cfg.obs_dim] - s).abs().max().item()
    assert diff < 1e-4, (
        f"fix_mask not holding: max |traj[:,0,:obs_dim] - s_norm| = {diff:.2e} >= 1e-4"
    )


# ── Unit tests — ordering ──────────────────────────────────────────────────────

def test_scores_descending(small_expansion, sample_state):
    """Scores must be sorted descending (best candidate first)."""
    result = small_expansion.expand(sample_state)
    scores = result.scores
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], (
            f"scores not descending at index {i}: {scores[i].item():.4f} < "
            f"{scores[i+1].item():.4f}"
        )


def test_best_traj_alias(small_expansion, sample_state):
    """best_traj must be the same tensor as trajs[0]."""
    result = small_expansion.expand(sample_state)
    assert torch.equal(result.best_traj, result.trajs[0])


def test_best_score_is_float(small_expansion, sample_state):
    """best_score must be a Python float, not a tensor."""
    result = small_expansion.expand(sample_state)
    assert isinstance(result.best_score, float)


# ── Unit tests — stochasticity ─────────────────────────────────────────────────

def test_planner_stochastic(small_expansion, sample_state):
    """Two expand() calls from the same start state must yield different trajs.

    This is a prerequisite for MCTS to produce diverse branches.  If the planner
    is deterministic (planner_self_l2 == 0), tree expansion degenerates.
    """
    torch.manual_seed(1)
    r1 = small_expansion.expand(sample_state)
    # No re-seeding — RNG state advances, so the next call differs
    r2 = small_expansion.expand(sample_state)
    # Fix_mask means position-0 is always equal; compare positions 1 onward
    are_equal = torch.allclose(r1.trajs[:, 1:, :], r2.trajs[:, 1:, :], atol=1e-6)
    assert not are_equal, (
        "Two sequential expand() calls returned identical trajectories — "
        "planner appears deterministic.  Check torch.randn_like seeding."
    )


# ── Unit tests — gradients ────────────────────────────────────────────────────

def test_no_grad_on_trajs(small_expansion, sample_state):
    """Output trajectories must not carry a gradient graph."""
    result = small_expansion.expand(sample_state)
    assert not result.trajs.requires_grad


def test_no_grad_on_scores(small_expansion, sample_state):
    """Output scores must not carry a gradient graph."""
    result = small_expansion.expand(sample_state)
    assert not result.scores.requires_grad


# ── Unit tests — device ───────────────────────────────────────────────────────

def test_output_on_correct_device(small_expansion, sample_state):
    """All output tensors must be on the device specified in config."""
    result = small_expansion.expand(sample_state)
    expected = torch.device(SMALL_CFG.device)
    assert result.trajs.device.type == expected.type
    assert result.scores.device.type == expected.type


# ── Unit tests — edge cases ───────────────────────────────────────────────────

def test_single_candidate(sample_state):
    """K=1 must work: shapes (1, H, D) and (1,)."""
    from cleandiffuser.diffusion import ContinuousDiffusionSDE
    from cleandiffuser.nn_diffusion import DiT1d
    from cleandiffuser.utils import DVHorizonCritic

    obs_dim, horizon = 8, 4
    cfg = ExpansionConfig(
        K=1, horizon=horizon, obs_dim=obs_dim, planner_dim=obs_dim,
        solver="ddim", sample_steps=2, temperature=1.0,
        use_ema=False, device=DEVICE,
    )
    nn_diff = DiT1d(obs_dim, emb_dim=32, d_model=64, n_heads=2, depth=2,
                    timestep_emb_type="fourier")
    fix_mask = torch.zeros((horizon, obs_dim))
    fix_mask[0] = 1.0
    planner = ContinuousDiffusionSDE(
        nn_diff, nn_condition=None, fix_mask=fix_mask,
        loss_weight=torch.ones((horizon, obs_dim)),
        ema_rate=0.9999, device=DEVICE, predict_noise=True, noise_schedule="linear",
    )
    planner.eval()
    critic = DVHorizonCritic(obs_dim, emb_dim=32, d_model=64, n_heads=2, depth=1).to(DEVICE)
    critic.eval()

    exp = PlannerExpansion(planner, critic, cfg)
    result = exp.expand(sample_state)

    assert result.trajs.shape == (1, horizon, obs_dim)
    assert result.scores.shape == (1,)


def test_bad_input_shape_raises(small_expansion):
    """expand() must raise ValueError if s_norm has wrong shape."""
    bad = torch.zeros(SMALL_CFG.obs_dim + 1)
    with pytest.raises(ValueError, match="s_norm must have shape"):
        small_expansion.expand(bad)


def test_bad_input_2d_raises(small_expansion):
    """expand() must reject a batched (1, obs_dim) input — must be 1-D."""
    bad = torch.zeros(1, SMALL_CFG.obs_dim)
    with pytest.raises(ValueError, match="s_norm must have shape"):
        small_expansion.expand(bad)


# ── Unit tests — immutability of config ───────────────────────────────────────

def test_config_frozen():
    """ExpansionConfig must be frozen — accidental mutation should raise."""
    cfg = ExpansionConfig(
        K=5, horizon=4, obs_dim=4, planner_dim=4,
        solver="ddim", sample_steps=2, temperature=1.0,
        use_ema=False, device="cpu",
    )
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        cfg.K = 10  # type: ignore[misc]


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
def real_expansion():
    """PlannerExpansion backed by the real maze2d-umaze-v1 checkpoint."""
    import torch
    from cleandiffuser.diffusion import ContinuousDiffusionSDE
    from cleandiffuser.nn_diffusion import DiT1d
    from cleandiffuser.utils import DVHorizonCritic

    PROD_CFG = ExpansionConfig(
        K=50,
        horizon=32,
        obs_dim=4,
        planner_dim=4,
        solver="ddim",
        sample_steps=20,
        temperature=1.0,
        use_ema=True,
        device="cpu",   # CPU for portability in tests
    )

    nn_diff = DiT1d(
        PROD_CFG.planner_dim, emb_dim=128,
        d_model=256, n_heads=4, depth=2,
        timestep_emb_type="fourier",
    )
    fix_mask = torch.zeros((PROD_CFG.horizon, PROD_CFG.planner_dim))
    fix_mask[0, : PROD_CFG.obs_dim] = 1.0

    planner = ContinuousDiffusionSDE(
        nn_diff, nn_condition=None,
        fix_mask=fix_mask,
        loss_weight=torch.ones((PROD_CFG.horizon, PROD_CFG.planner_dim)),
        ema_rate=0.9999, device="cpu",
        predict_noise=True, noise_schedule="linear",
    )
    planner.load(os.path.join(CKPT_DIR, "planner_ckpt_1000000.pt"))
    planner.eval()

    critic = DVHorizonCritic(
        PROD_CFG.planner_dim, emb_dim=128,
        d_model=256, n_heads=4, depth=2, norm_type="pre",
    ).to("cpu")
    critic_ckpt = torch.load(
        os.path.join(CKPT_DIR, "critic_ckpt_1000000.pt"),
        map_location="cpu",
    )
    critic.load_state_dict(critic_ckpt["critic"])
    critic.eval()

    return PlannerExpansion(planner, critic, PROD_CFG), PROD_CFG


@integration
@requires_checkpoint
def test_integration_fix_mask_tight(real_expansion):
    """fix_mask must clamp position-0 to s_norm within 1e-4 (Phase 1 assertion)."""
    exp, cfg = real_expansion
    torch.manual_seed(0)
    s_norm = torch.zeros(cfg.obs_dim)  # normalised origin
    result = exp.expand(s_norm)
    diff = (result.trajs[:, 0, : cfg.obs_dim] - s_norm).abs().max().item()
    assert diff < 1e-4, f"fix_mask violated: max-abs = {diff:.2e}"


@integration
@requires_checkpoint
def test_integration_score_range(real_expansion):
    """Critic scores on generated plans should lie in [-1.5, 1.5].

    The critic is trained to output MC returns normalised to [-1, 1].
    A small margin accounts for floating-point drift outside training distribution.
    Phase 1 confirmed scores in [0.50, 0.97] for generated plans from held-out
    states — all well within this bound.
    """
    exp, cfg = real_expansion
    torch.manual_seed(0)
    s_norm = torch.zeros(cfg.obs_dim)
    result = exp.expand(s_norm)
    assert result.scores.min().item() >= -1.5, (
        f"scores below -1.5: min = {result.scores.min().item():.4f}"
    )
    assert result.scores.max().item() <= 1.5, (
        f"scores above 1.5: max = {result.scores.max().item():.4f}"
    )


@integration
@requires_checkpoint
def test_integration_stochastic_real_model(real_expansion):
    """Real planner must produce diverse trajectories (planner_self_l2 > 0.05).

    Phase 1 confirmed mean planner_self_l2 = 0.305 with min = 0.042 across
    100 held-out states.  A single-call check at threshold 0.05 is conservative.
    """
    exp, cfg = real_expansion
    torch.manual_seed(1)
    s_norm = torch.zeros(cfg.obs_dim)
    r1 = exp.expand(s_norm)
    r2 = exp.expand(s_norm)
    # Compare positions 1 onward (position-0 is always fixed by fix_mask)
    l2 = (r1.trajs[:, 1:, :] - r2.trajs[:, 1:, :]).norm(dim=-1).mean().item()
    assert l2 > 0.05, (
        f"planner_self_l2 = {l2:.4f} — planner appears near-deterministic. "
        "MCTS branching requires stochastic sampling."
    )


@integration
@requires_checkpoint
def test_integration_shapes_production(real_expansion):
    """Production config must produce (50, 32, 4) trajs and (50,) scores."""
    exp, cfg = real_expansion
    torch.manual_seed(0)
    s_norm = torch.zeros(cfg.obs_dim)
    result = exp.expand(s_norm)
    assert result.trajs.shape == (50, 32, 4)
    assert result.scores.shape == (50,)
