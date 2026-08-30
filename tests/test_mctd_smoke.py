"""tests/test_mctd_smoke.py

End-to-end smoke tests for MCTDPlanner (mcts/mctd_planner.py) — the whole loop
wired together (tree + block/jumpy denoise + guidance + geometric verify + output
selection). The first two run torch-only with a RANDOM-WEIGHT planner and an
identity normalizer (no checkpoint, no gym/d4rl), so they check "everything is
connected and returns sane, finite, correctly-shaped output — no NaNs, no garbage
values" anywhere torch imports. The last one is gated on a real DF checkpoint +
d4rl and skips off the training box.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

torch = pytest.importorskip("torch")
try:
    from mcts.df_model import DFPlanner
    from mcts.mctd_planner import MCTDConfig, MCTDPlanner
except Exception as exc:
    pytest.skip(f"MCTD/DFPlanner unavailable: {exc!r}", allow_module_level=True)

D, K, T = 4, 8, 6


class _IdentityNormalizer:
    """normalize/unnormalize are the identity; mean/std present for
    mcts.specs.normalize_goal_xy (which reads .mean and calls .normalize)."""
    def __init__(self, D):
        self.mean = np.zeros(D, dtype=np.float32)
        self.std = np.ones(D, dtype=np.float32)

    def normalize(self, x):
        return np.asarray(x, dtype=np.float32)

    def unnormalize(self, x):
        return np.asarray(x, dtype=np.float32)


def _planner():
    torch.manual_seed(0)
    return DFPlanner(in_dim=D, K=K, d_model=32, n_heads=2, depth=1, emb_dim=16,
                     device="cpu")


def _mctd(goal_radius, warp_threshold=None):
    cfg = MCTDConfig(guidance_scales=(0.0, 1.0), n_depths=2, max_search_num=8,
                     num_tries_for_bad_plans=2, skip_level_steps=2)
    return MCTDPlanner(df_planner=_planner(), normalizer=_IdentityNormalizer(D),
                       family="maze2d", obs_dim=D, H=T, cfg=cfg,
                       env_cfg=dict(pos_dims=(0, 1), goal_radius=goal_radius,
                                    warp_threshold=warp_threshold),
                       device="cpu")


def _assert_well_formed(out, terminal_depth=2):
    assert out["plan_norm"].shape == (T, D)
    assert np.all(np.isfinite(out["plan_norm"]))            # no NaN/inf plan
    assert isinstance(out["solved"], bool)
    assert out["info"] in ("Achieved", "NotReached", "Failed")
    assert 0.0 <= out["value"] <= 1.0 + 1e-9
    assert out["n_search"] >= 1
    assert 0 <= out["max_depth"] <= terminal_depth
    assert out["terminal_depth"] == terminal_depth


def test_plan_reaches_and_is_well_formed_with_large_radius():
    # a huge goal radius forces "Achieved" quickly -> exercises the solved path
    out = _mctd(goal_radius=1e9).plan(start_norm=np.zeros(D, np.float32),
                                      goal_raw=np.array([3.0, 0.0]), seed=0)
    _assert_well_formed(out)
    assert out["solved"] and out["info"] == "Achieved"


def test_plan_returns_best_miss_with_tiny_radius():
    # a tiny radius means nothing reaches -> exercises the not-reached fallback
    out = _mctd(goal_radius=1e-6).plan(start_norm=np.zeros(D, np.float32),
                                       goal_raw=np.array([3.0, 0.0]), seed=1)
    _assert_well_formed(out)
    assert not out["solved"]
    assert out["info"] in ("NotReached", "Failed")


def test_kitchen_family_is_refused():
    with pytest.raises(ValueError):
        MCTDPlanner(df_planner=_planner(), normalizer=_IdentityNormalizer(D),
                    family="kitchen", obs_dim=D, H=T)


def test_determinism_same_seed_same_plan():
    a = _mctd(goal_radius=1e9).plan(np.zeros(D, np.float32),
                                    np.array([3.0, 0.0]), seed=5)
    b = _mctd(goal_radius=1e9).plan(np.zeros(D, np.float32),
                                    np.array([3.0, 0.0]), seed=5)
    assert np.allclose(a["plan_norm"], b["plan_norm"])
    assert a["n_search"] == b["n_search"]


def _fake_critic():
    """A stand-in for the DV critic: scores a window by the mean of channel 0
    over time. Deterministic, torch-only, no cleandiffuser needed."""
    def crit(x):                          # x: (N, H, D) -> (N, 1)
        return x[..., 0].mean(dim=1, keepdim=True)
    return crit


def test_critic_value_mode_well_formed():          # Way 4c
    p = _planner()
    cfg = MCTDConfig(guidance_scales=(0.0, 1.0), n_depths=2, max_search_num=8,
                     num_tries_for_bad_plans=2, skip_level_steps=2)
    planner = MCTDPlanner(df_planner=p, normalizer=_IdentityNormalizer(D),
                          family="maze2d", obs_dim=D, H=T, cfg=cfg,
                          env_cfg=dict(pos_dims=(0, 1), goal_radius=1.0,
                                       warp_threshold=None),
                          device="cpu", value_mode="critic", critic=_fake_critic())
    out = planner.plan(np.zeros(D, np.float32), np.array([3.0, 0.0]), seed=0)
    assert out["plan_norm"].shape == (T, D)
    assert np.all(np.isfinite(out["plan_norm"]))
    assert out["n_search"] >= 1
    # critic mode has no reach concept -> never "Achieved"
    assert out["info"] in ("NotReached", "Failed") and out["solved"] is False


def test_critic_mode_requires_critic():
    with pytest.raises(ValueError):
        MCTDPlanner(df_planner=_planner(), normalizer=_IdentityNormalizer(D),
                    family="maze2d", obs_dim=D, H=T, value_mode="critic", critic=None)


def test_guided_bon_well_formed():                 # Way 4b
    from mcts.mctd_loop import GuidedBoNPlanner
    models = dict(df_planner=_planner(), critic=_fake_critic(),
                  normalizer=_IdentityNormalizer(D), device="cpu", H=T, obs_dim=D)
    gb = GuidedBoNPlanner(models, family="maze2d", guidance_scales=(0.0, 1.0),
                          k_per=3)
    out = gb.plan(np.zeros(D, np.float32), np.array([3.0, 0.0]), seed=0)
    assert out["plan_norm"].shape == (T, D)
    assert np.all(np.isfinite(out["plan_norm"]))
    assert out["n_search"] == 2 * 3                # len(menu) * k_per
    assert out["max_depth"] == 0                   # flat, no tree


@pytest.mark.parametrize("env_name", ["maze2d-large-v1"])
def test_real_checkpoint_single_plan(env_name):
    """Gated: needs the DF checkpoint + d4rl. Skips off the training box."""
    try:
        from mcts.mcts_loop import load_models
        from mcts.specs import get_goal
        models = load_models(env_name, df_ckpt="final")
    except Exception as exc:
        pytest.skip(f"real-env smoke unavailable: {exc!r}")
    import gym  # noqa: F401
    planner = MCTDPlanner.from_models(models, MCTDConfig(max_search_num=6))
    env = models["env_single"]
    obs = env.reset()
    goal = get_goal(env)
    s_norm = models["normalizer"].normalize(np.asarray(obs)[None])[0].astype(np.float32)
    out = planner.plan(s_norm, np.asarray(goal, dtype=np.float64), seed=0)
    assert out["plan_norm"].shape == (models["H"], models["obs_dim"])
    assert np.all(np.isfinite(out["plan_norm"]))
    assert 0.0 <= out["value"] <= 1.0 + 1e-9
    print(f"[{env_name}] MCTD: solved={out['solved']} info={out['info']} "
          f"value={out['value']:.3f} depth={out['max_depth']}/{out['terminal_depth']} "
          f"searches={out['n_search']} nodes={out['n_nodes']}")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v) and "real_checkpoint" not in k]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
