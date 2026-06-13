"""tests/test_collate_mcts.py

Pure-stdlib tests for scripts/collate_mcts.py: the exact McNemar test (hand-computed
cases), the binomial reach SEM, compute accounting, and tolerant JSON loading across
the old (aggregates-only) and new (per-rollout vectors) result schemas.
"""
import json
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from collate_mcts import (binom_err_pct, candidates_per_step, load_rows,
                          mcnemar_exact, pairing_check, print_mcnemar)


def _pairs(fixes, breaks, both_reach=10, both_miss=2):
    """Build paired 0/1 vectors with the given discordance pattern."""
    a = [0] * fixes + [1] * breaks + [1] * both_reach + [0] * both_miss
    b = [1] * fixes + [0] * breaks + [1] * both_reach + [0] * both_miss
    return a, b


def test_mcnemar_pure_fixes():
    a, b = _pairs(fixes=3, breaks=0)
    f, br, p = mcnemar_exact(a, b)
    assert (f, br) == (3, 0)
    assert abs(p - 0.25) < 1e-12          # 2 * C(3,0) * 0.5^3


def test_mcnemar_no_discordance():
    a, b = _pairs(fixes=0, breaks=0)
    f, br, p = mcnemar_exact(a, b)
    assert (f, br) == (0, 0) and p == 1.0


def test_mcnemar_capped_at_one():
    a, b = _pairs(fixes=1, breaks=1)
    _, _, p = mcnemar_exact(a, b)
    assert p == 1.0                        # 2 * (C(2,0)+C(2,1)) * 0.25 = 1.5 → capped


def test_mcnemar_hand_computed_14_2():
    a, b = _pairs(fixes=14, breaks=2)
    f, br, p = mcnemar_exact(a, b)
    expect = 2.0 * (math.comb(16, 0) + math.comb(16, 1) + math.comb(16, 2)) * 0.5 ** 16
    assert (f, br) == (14, 2)
    assert abs(p - expect) < 1e-15


def test_mcnemar_symmetric_p():
    a, b = _pairs(fixes=5, breaks=2)
    f1, b1, p1 = mcnemar_exact(a, b)
    f2, b2, p2 = mcnemar_exact(b, a)       # swapped: fixes/breaks swap, same p
    assert (f1, b1) == (b2, f2)
    assert abs(p1 - p2) < 1e-15


def test_mcnemar_length_mismatch_raises():
    try:
        mcnemar_exact([1, 0], [1])
    except ValueError:
        return
    raise AssertionError("expected ValueError for unequal vector lengths")


def test_binom_err():
    assert abs(binom_err_pct(0.5, 100) - 5.0) < 1e-12
    assert binom_err_pct(0.0, 25) == 0.0
    assert math.isnan(binom_err_pct(0.5, 0))


def test_candidates_per_step():
    assert candidates_per_step({"k_mcts": 16, "budget": 16}, "mcts") == 272
    assert candidates_per_step({"k_mcss": 50}, "mcss") == 50


def _write(tmp, name, payload):
    path = os.path.join(tmp, name)
    with open(path, "w") as f:
        json.dump(payload, f)
    return path


def test_load_rows_old_and_new_schemas():
    with tempfile.TemporaryDirectory() as tmp:
        old = _write(tmp, "old.json", dict(
            env="antmaze-large-diverse-v2", seed=0, k_mcss=272,
            results=dict(mcss=dict(method="mcss", n_rollouts=25, reach_pct=84.0,
                                   norm_mean=84.0, norm_err=7.3, wall_s=5896.0))))
        succ = [1] * 24 + [0]
        new = _write(tmp, "new.json", dict(
            env="antmaze-large-diverse-v2", seed=0, k_mcts=16, budget=16,
            results=dict(mcts=dict(method="mcts", n_rollouts=25, reach_pct=96.0,
                                   reach_err=3.9, norm_mean=96.0, norm_err=3.9,
                                   wall_s=5579.0, success=succ))))
        rows = load_rows([old, new])
    assert len(rows) == 2
    by_m = {r["method"]: r for r in rows}
    assert by_m["mcss"]["success"] is None          # legacy: no vector
    assert by_m["mcss"]["err"] == 7.3               # legacy fallback: norm_err
    assert by_m["mcts"]["success"] == succ
    expect_err = binom_err_pct(24 / 25, 25)         # vector-derived binomial SEM
    assert abs(by_m["mcts"]["err"] - expect_err) < 1e-12
    assert by_m["mcts"]["cands"] == 272


def test_pairing_check_goals_match():
    g = [[32.41, 25.27], [33.21, 25.19]]
    a = dict(goals=[list(x) for x in g], starts=[[0.01, -0.02], [0.03, 0.00]])
    b = dict(goals=[list(x) for x in g], starts=[[-0.05, 0.04], [0.02, -0.09]])
    ok, note = pairing_check(a, b)
    assert ok and note.startswith("goals match")
    assert "start jitter" in note and "suspect" not in note   # ±0.1 reset noise is fine


def test_pairing_check_goals_differ():
    a = dict(goals=[[32.41, 25.27], [33.21, 25.19]])
    b = dict(goals=[[32.41, 25.27], [20.00, 5.00]])           # index 1: different scenario
    ok, note = pairing_check(a, b)
    assert not ok and "1/2" in note


def test_pairing_check_unverified_when_missing():
    ok, note = pairing_check(dict(goals=None), dict(goals=[[1, 2]]))
    assert ok and "unverified" in note


def test_pairing_check_flags_large_start_divergence():
    g = [[32.41, 25.27]]
    a = dict(goals=g, starts=[[0.0, 0.0]])
    b = dict(goals=g, starts=[[3.0, 0.0]])                    # a different cell entirely
    ok, note = pairing_check(a, b)
    assert ok and "suspect" in note


def test_child_index_label_and_depth_field():
    with tempfile.TemporaryDirectory() as tmp:
        p1 = _write(tmp, "l1.json", dict(
            env="antmaze", seed=0, k_mcts=16, budget=16, child_index=1,
            results=dict(mcts=dict(n_rollouts=4, reach_pct=75.0, wall_s=1.0,
                                   tree_depth_mean=3.4, success=[1, 1, 1, 0]))))
        p4 = _write(tmp, "l4.json", dict(
            env="antmaze", seed=0, k_mcts=16, budget=16, child_index=4,
            results=dict(mcts=dict(n_rollouts=4, reach_pct=75.0, wall_s=1.0,
                                   success=[1, 1, 1, 0]))))
        old = _write(tmp, "old.json", dict(           # pre-cidx JSONs: no child_index key
            env="antmaze", seed=0, k_mcts=16, budget=8,
            results=dict(mcts=dict(n_rollouts=4, reach_pct=50.0, wall_s=1.0))))
        rows = load_rows([p1, p4, old])
    labels = sorted(r["label"] for r in rows)
    assert labels == ["b16", "b16L4", "b8"]           # cidx=1 and absent → plain; cidx=4 → suffixed
    by_label = {r["label"]: r for r in rows}
    assert by_label["b16"]["depth"] == 3.4
    assert by_label["b16L4"]["depth"] is None         # tolerant when not recorded


def test_print_mcnemar_pairs_and_skips():
    """Same (env, seed): equal-length pair runs; unequal-length pair is skipped."""
    a, b = _pairs(fixes=3, breaks=0)
    rows = [
        dict(env="antmaze", seed=0, method="mcss", label="k272", cands=272,
             reach=100 * sum(a) / len(a), err=0.0, n=len(a), wall=0.0,
             success=a, file="a.json"),
        dict(env="antmaze", seed=0, method="mcts", label="b16", cands=272,
             reach=100 * sum(b) / len(b), err=0.0, n=len(b), wall=0.0,
             success=b, file="b.json"),
        dict(env="antmaze", seed=0, method="mcts", label="b8", cands=144,
             reach=50.0, err=0.0, n=4, wall=0.0,
             success=[1, 0, 1, 0], file="c.json"),   # n mismatch → skipped
    ]
    print_mcnemar(rows)                              # must not raise


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
