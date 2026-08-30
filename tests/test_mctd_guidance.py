"""tests/test_mctd_guidance.py

Tests for the MCTD goal guidance (mcts/mctd_guidance.py). The important property
is DIRECTION: a gradient-ascent step on value(x) must move the plan's future
tokens TOWARD the goal (value = -distance, so ascending reduces distance). If the
sign were wrong the guidance meta-action would push plans AWAY from the goal and
the whole search would be worse than unguided — a silent garbage mode. Also
checks the history token is excluded from the objective (its gradient is zero).

Needs only torch.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

torch = pytest.importorskip("torch")

from mcts.mctd_guidance import GoalGuide

D, T, N = 5, 8, 4
POS = (0, 1)


def test_value_shape_and_history_excluded():
    guide = GoalGuide(goal_norm=[3.0, 0.0], pos_dims=POS)
    x = torch.zeros(N, T, D, requires_grad=True)
    v = guide.value(x, torch.zeros(N, T, dtype=torch.long))
    assert v.shape == (N,)
    v.sum().backward()
    # token 0 is the clamped start/history -> not in the objective -> zero grad
    assert torch.allclose(x.grad[:, 0], torch.zeros(N, D), atol=0)
    # future tokens DO get gradient
    assert x.grad[:, 1:, POS[0]].abs().sum() > 0


def test_ascent_step_moves_future_tokens_toward_goal():
    goal = torch.tensor([5.0, -2.0])
    guide = GoalGuide(goal_norm=goal.tolist(), pos_dims=POS)
    x = torch.zeros(N, T, D)                    # plan starts far from goal
    x = x.clone().requires_grad_(True)
    guide.value(x, torch.zeros(N, T, dtype=torch.long)).sum().backward()
    stepped = x.detach() + 0.5 * x.grad         # gradient ASCENT on value

    def goal_dist(z):
        p = z[:, 1:, POS]                       # future tokens' positions
        return torch.linalg.norm(p - goal.view(1, 1, -1), dim=-1).mean()

    assert goal_dist(stepped) < goal_dist(x.detach())


def test_guidance_scale_zero_is_identity_in_sampler_contract():
    # value(x) itself is scale-free; the SCALE multiplies the eps-shift in the
    # sampler (mcts/mctd_denoise.py). Here we just confirm value is finite and
    # deterministic so scaling it by 0 in the sampler is a clean no-op.
    guide = GoalGuide(goal_norm=[1.0, 1.0], pos_dims=POS)
    x = torch.randn(N, T, D)
    v1 = guide.value(x, torch.zeros(N, T, dtype=torch.long))
    v2 = guide.value(x, torch.zeros(N, T, dtype=torch.long))
    assert torch.allclose(v1, v2) and torch.isfinite(v1).all()


def test_goal_dim_mismatch_raises():
    with pytest.raises(ValueError):
        GoalGuide(goal_norm=[1.0, 2.0, 3.0], pos_dims=(0, 1))   # 3 dims vs 2


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
