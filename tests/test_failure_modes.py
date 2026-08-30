"""tests/test_failure_modes.py

Pure-stdlib unit tests for the Tier-1 failure-mode classifier
(`mcts.failure_modes`). No numpy/torch/d4rl — runs on the local Windows box:

    python -m unittest tests.test_failure_modes -v

Each test builds a synthetic per-step BFS-distance curve (and, where relevant,
body-state and candidate-pool numbers) whose intended failure mode is obvious,
then asserts the classifier recovers it. These curves are the contract: if a
threshold is retuned, the test names say which behaviour must still hold.
"""
import math
import unittest

from mcts.failure_modes import (
    INF, ClassifierConfig, FailureRecord, classify_failure, progress_features,
    pose_collapse_step, sustained_collapse_onset, candidate_pool_quality, tally,
    FELL_OVER, GOAL_RADIUS_ARTIFACT, WRONG_TURN, NO_GOOD_PLAN, OSCILLATION,
    TIMEOUT_ON_TRACK, UNREACHABLE_FAR, OFF_GRAPH,
    POOL_GOOD_NOT_PICKED, POOL_NO_GOOD, POOL_GOOD_PICKED,
    CRITIC_FIXABLE, CRITIC_IMMUNE,
)

CFG = ClassifierConfig()


def upright_ok(n):
    return [1.0] * n


def height_ok(n):
    return [0.55] * n


class TestProgressFeatures(unittest.TestCase):
    def test_monotone_descent(self):
        d = [10.0, 8.0, 6.0, 4.0, 2.0]
        f = progress_features(d)
        self.assertEqual(f.start_dist, 10.0)
        self.assertEqual(f.end_dist, 2.0)
        self.assertEqual(f.min_dist, 2.0)
        self.assertEqual(f.argmin_step, 4)
        self.assertAlmostEqual(f.net_progress, 8.0)
        self.assertEqual(f.backslide, 0.0)
        self.assertEqual(f.reversals, 0)
        self.assertGreater(f.tail_closing, 0.0)

    def test_wrong_turn_shape(self):
        # down to 2 at step 4, then back up to 9 and stays high
        d = [10, 7, 5, 3, 2, 4, 6, 8, 9, 9]
        f = progress_features([float(x) for x in d])
        self.assertEqual(f.min_dist, 2.0)
        self.assertEqual(f.argmin_step, 4)
        self.assertAlmostEqual(f.gained, 8.0)        # 10 -> 2
        self.assertAlmostEqual(f.backslide, 7.0)     # 9 - 2
        self.assertLessEqual(f.tail_closing, 0.0)    # not closing at the tail

    def test_off_graph_handling(self):
        d = [10.0, INF, 8.0, INF, 6.0]
        f = progress_features(d)
        self.assertAlmostEqual(f.off_graph_frac, 2 / 5)
        self.assertEqual(f.start_dist, 10.0)
        self.assertEqual(f.end_dist, 6.0)
        self.assertFalse(f.all_off_graph)

    def test_all_off_graph(self):
        f = progress_features([INF, INF, INF])
        self.assertTrue(f.all_off_graph)
        self.assertEqual(f.argmin_step, -1)

    def test_reversal_count(self):
        # up-down-up-down sawtooth
        d = [5, 6, 5, 6, 5, 6]
        f = progress_features([float(x) for x in d])
        self.assertGreaterEqual(f.reversals, 4)


class TestPoseCollapse(unittest.TestCase):
    def test_detects_low_uprightness(self):
        up = [1.0, 1.0, 0.2, 0.1]      # toppled at step 2
        self.assertEqual(pose_collapse_step(up, height_ok(4)), 2)

    def test_detects_low_height(self):
        h = [0.55, 0.55, 0.2]          # collapsed at step 2
        self.assertEqual(pose_collapse_step(upright_ok(3), h), 2)

    def test_no_collapse(self):
        self.assertIsNone(pose_collapse_step(upright_ok(5), height_ok(5)))

    def test_empty_curves(self):
        self.assertIsNone(pose_collapse_step([], []))

    def test_transient_tilt_not_sustained(self):
        # a single gait lean that recovers is NOT a fall
        up = [1, 1, 0.2, 1, 1, 1, 1, 1, 1, 1]
        self.assertIsNone(sustained_collapse_onset([float(x) for x in up], height_ok(10)))

    def test_sustained_collapse_onset(self):
        # down from step 3 through the end -> onset 3
        up = [1, 1, 1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        self.assertEqual(sustained_collapse_onset([float(x) for x in up], height_ok(10)), 3)


class TestCandidatePool(unittest.TestCase):
    def test_good_not_picked(self):
        # current at 10; a candidate reaches 5 (good) but chosen ends at 11 (away)
        v, ev = candidate_pool_quality([5.0, 9.0, 12.0], chosen_dist=11.0, cur_dist=10.0)
        self.assertEqual(v, POOL_GOOD_NOT_PICKED)
        self.assertEqual(ev["best_dist"], 5.0)

    def test_no_good_candidate(self):
        # nothing gets meaningfully closer than 10
        v, _ = candidate_pool_quality([10.0, 11.0, 12.0], chosen_dist=11.0, cur_dist=10.0)
        self.assertEqual(v, POOL_NO_GOOD)

    def test_good_picked(self):
        v, _ = candidate_pool_quality([5.0, 9.0, 12.0], chosen_dist=5.0, cur_dist=10.0)
        self.assertEqual(v, POOL_GOOD_PICKED)

    def test_all_infinite_candidates(self):
        v, _ = candidate_pool_quality([INF, INF], chosen_dist=INF, cur_dist=10.0)
        self.assertEqual(v, POOL_NO_GOOD)


class TestClassify(unittest.TestCase):
    def test_goal_radius_artifact(self):
        # came within the TRUE reward radius (0.35 < 0.5 world units) yet success=0;
        # artifact is first priority so it wins over the otherwise-wrong-turn shape
        d = [12, 9, 6, 4, 3, 5, 8, 10, 11, 11]
        rec = FailureRecord(dist=[float(x) for x in d], upright=upright_ok(10),
                            height=height_ok(10), success=False, min_world_dist=0.35)
        mode, _ = classify_failure(rec)
        self.assertEqual(mode, GOAL_RADIUS_ARTIFACT)

    def test_near_goal_outside_radius_not_artifact(self):
        # reached the goal CELL region but stayed outside the 0.5 reward radius (F1):
        # must surface as a REAL failure (wrong turn), not be buried as immune artifact
        d = [12, 9, 6, 4, 3, 5, 8, 10, 11, 11]
        rec = FailureRecord(dist=[float(x) for x in d], upright=upright_ok(10),
                            height=height_ok(10), success=False, min_world_dist=1.8,
                            junction_cand_dists=[1.0, 2.0, 9.0], junction_chosen_dist=8.0)
        mode, _ = classify_failure(rec)
        self.assertNotEqual(mode, GOAL_RADIUS_ARTIFACT)
        self.assertEqual(mode, WRONG_TURN)

    def test_missing_world_dist_skips_artifact(self):
        # no executed xy -> cannot test the radius -> fall through, never call it immune
        d = [10, 6, 3, 1, 0.5, 0.5]
        rec = FailureRecord(dist=[float(x) for x in d], upright=upright_ok(6),
                            height=height_ok(6), success=False, min_world_dist=None)
        mode, _ = classify_failure(rec)
        self.assertNotEqual(mode, GOAL_RADIUS_ARTIFACT)

    def test_fell_over_before_bad(self):
        # topples at step 1, before reaching the closest point
        d = [10, 9, 9, 10, 11]
        up = [1.0, 0.1, 0.0, 0.0, 0.0]
        rec = FailureRecord(dist=[float(x) for x in d], upright=up,
                            height=height_ok(5), success=False)
        mode, _ = classify_failure(rec)
        self.assertEqual(mode, FELL_OVER)

    def test_wrong_turn_good_not_picked(self):
        d = [12, 9, 6, 4, 3, 5, 8, 10, 11, 11]   # advance then backslide; junction at step 4 (min=3)
        rec = FailureRecord(dist=[float(x) for x in d], upright=upright_ok(10),
                            height=height_ok(10), success=False,
                            junction_cand_dists=[1.0, 2.0, 9.0],  # a candidate reaches 1 (goalward)
                            junction_chosen_dist=8.0)             # but chose one heading away
        mode, ev = classify_failure(rec)
        self.assertEqual(mode, WRONG_TURN)
        self.assertEqual(ev["pool_verdict"], POOL_GOOD_NOT_PICKED)
        self.assertIn(mode, CRITIC_FIXABLE)

    def test_wrong_turn_no_good_plan(self):
        d = [12, 9, 6, 4, 3, 5, 8, 10, 11, 11]
        rec = FailureRecord(dist=[float(x) for x in d], upright=upright_ok(10),
                            height=height_ok(10), success=False,
                            junction_cand_dists=[3.0, 4.0, 9.0],  # nothing closer than ~min
                            junction_chosen_dist=4.0)
        mode, ev = classify_failure(rec)
        self.assertEqual(mode, NO_GOOD_PLAN)
        self.assertIn(mode, CRITIC_IMMUNE)

    def test_good_picked_routes_to_execution(self):
        # advanced then backslid, but the chosen candidate WAS goalward -> downstream/exec
        d = [12, 9, 6, 4, 3, 5, 8, 10, 11, 11]
        rec = FailureRecord(dist=[float(x) for x in d], upright=upright_ok(10),
                            height=height_ok(10), success=False,
                            junction_cand_dists=[1.0, 2.0, 9.0],
                            junction_chosen_dist=1.0)             # picked the goalward one
        mode, _ = classify_failure(rec)
        self.assertEqual(mode, FELL_OVER)

    def test_transient_tilt_with_wrong_turn_is_wrong_turn(self):
        # early gait lean that recovers + a goalward candidate unpicked at the junction:
        # recalibration -> WRONG_TURN, NOT FELL_OVER (transient leans aren't falls, and
        # the ranking signal pre-empts the pose guard)
        d = [12, 9, 6, 4, 3, 5, 8, 10, 11, 11]
        up = [1, 0.2, 1, 1, 1, 1, 1, 1, 1, 1]
        rec = FailureRecord(dist=[float(x) for x in d], upright=up, height=height_ok(10),
                            success=False, junction_cand_dists=[1.0, 2.0, 9.0],
                            junction_chosen_dist=8.0)
        mode, _ = classify_failure(rec)
        self.assertEqual(mode, WRONG_TURN)

    def test_oscillation(self):
        d = [8, 9, 8, 9, 8, 9, 8, 9, 8, 9]       # many reversals, no net progress
        rec = FailureRecord(dist=[float(x) for x in d], upright=upright_ok(10),
                            height=height_ok(10), success=False)
        mode, _ = classify_failure(rec)
        self.assertEqual(mode, OSCILLATION)

    def test_oscillation_is_immune(self):
        # F2: plain oscillation is NOT credited to the critic
        self.assertIn(OSCILLATION, CRITIC_IMMUNE)
        self.assertNotIn(OSCILLATION, CRITIC_FIXABLE)

    def test_value_side_oscillation_is_wrong_turn(self):
        # ping-pong, BUT a goalward candidate existed unpicked at the junction (F2):
        # the value-side of oscillation is the critic's to fix -> WRONG_TURN
        d = [8, 9, 8, 9, 8, 9, 8, 9, 8, 9]
        rec = FailureRecord(dist=[float(x) for x in d], upright=upright_ok(10),
                            height=height_ok(10), success=False,
                            junction_cand_dists=[6.0, 7.0, 9.0], junction_chosen_dist=9.0)
        mode, _ = classify_failure(rec)
        self.assertEqual(mode, WRONG_TURN)

    def test_timeout_on_track(self):
        # still closing at the tail, never gave ground back
        d = [40, 38, 36, 34, 32, 30, 28, 26, 24, 22]
        rec = FailureRecord(dist=[float(x) for x in d], upright=upright_ok(10),
                            height=height_ok(10), success=False, is_far=False)
        mode, _ = classify_failure(rec)
        self.assertEqual(mode, TIMEOUT_ON_TRACK)

    def test_unreachable_far(self):
        d = [60, 58, 56, 54, 52, 50, 48, 46, 44, 42]
        rec = FailureRecord(dist=[float(x) for x in d], upright=upright_ok(10),
                            height=height_ok(10), success=False, is_far=True)
        mode, _ = classify_failure(rec)
        self.assertEqual(mode, UNREACHABLE_FAR)

    def test_off_graph(self):
        d = [10, INF, INF, INF, 9]              # mostly off-graph, no pose collapse
        rec = FailureRecord(dist=d, upright=upright_ok(5), height=height_ok(5),
                            success=False)
        mode, _ = classify_failure(rec)
        self.assertEqual(mode, OFF_GRAPH)

    def test_fixable_immune_partition_total(self):
        # sanity: the two attribution sets don't overlap
        self.assertEqual(CRITIC_FIXABLE & CRITIC_IMMUNE, frozenset())


class TestTally(unittest.TestCase):
    def test_counts_and_pct(self):
        rows = tally([WRONG_TURN, WRONG_TURN, FELL_OVER, TIMEOUT_ON_TRACK])
        top = rows[0]
        self.assertEqual(top[0], WRONG_TURN)
        self.assertEqual(top[1], 2)
        self.assertAlmostEqual(top[2], 50.0)


if __name__ == "__main__":
    unittest.main()
