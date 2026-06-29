"""mcts/failure_modes.py

Pure-Python (no numpy/torch) failure-mode classification for the closed-loop
sampler — the *scientific core* of the Tier-1 instrumentation.

Why torch-free
--------------
Like `mcts/value_forest.py`, the analysis logic here is intentionally free of
heavyweight deps: it consumes plain per-step lists of floats (the BFS-distance
curve, the body-state curves, the candidate-endpoint distances) and returns a
classification. That makes every decision rule unit-testable on the local box
(`tests/test_failure_modes.py`) with synthetic curves — no GPU, no d4rl. The
numpy/torch I/O that produces these curves lives in `mcts/instrument.py`.

What it decides
---------------
A failed rollout (the ant never touched the goal) is assigned ONE primary mode,
turning "20% failed" into a per-mode tally. Only some modes are within the
project's reach (a better critic): WRONG_TURN (ranking) and the value-side of
OSCILLATION. The rest — FELL_OVER (execution), NO_GOOD_PLAN (proposal/coverage),
TIMEOUT_ON_TRACK / UNREACHABLE_FAR (horizon), GOAL_RADIUS_ARTIFACT (measurement)
— are NOT critic-fixable and must be separated out before any "oracle solves N of
M" ceiling (Tier 2) is interpretable. That ordering is the whole point of Tier 1.

Distances are in BFS *cell* units (the oracle's native unit); an off-graph
executed state (in a wall / unreachable from the goal) is recorded as
`float('inf')` and handled explicitly (it usually signals a clip or a fall).

Oracle discipline: this module never imports the oracle and never sees an env —
it only consumes numbers already computed upstream. It is analysis, not an oracle
consumer, so it is safe to import anywhere.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

INF = float("inf")

# Mode names (single source — scripts import these so a typo can't desync tallies).
FELL_OVER = "FELL_OVER"                      # execution: torso collapsed before the path went bad
GOAL_RADIUS_ARTIFACT = "GOAL_RADIUS_ARTIFACT"  # measurement: reached the goal cell yet success=0
WRONG_TURN = "WRONG_TURN"                    # ranking (critic-fixable): a good option existed, unpicked
NO_GOOD_PLAN = "NO_GOOD_PLAN"                # proposal/coverage: no goalward candidate existed
OSCILLATION = "OSCILLATION"                  # ping-pong at a junction, net no progress
TIMEOUT_ON_TRACK = "TIMEOUT_ON_TRACK"        # horizon: still making net progress at the limit
UNREACHABLE_FAR = "UNREACHABLE_FAR"          # structural: maximal start distance, ran out of horizon
OFF_GRAPH = "OFF_GRAPH"                      # mostly off the maze graph without a clean pose collapse
UNCLASSIFIED = "UNCLASSIFIED"                # guard — features didn't match any rule

# Which modes a better critic (this project's lever) can plausibly move.
# Only WRONG_TURN is credited to the critic. Value-side oscillation (a goalward
# candidate existed at the junction but went unpicked) is RELABELLED to WRONG_TURN
# inside the classifier, so any residual OSCILLATION here is the policy/proposal
# kind and stays immune — otherwise the upper bound is inflated with execution
# ping-ponging the critic cannot fix (F2).
CRITIC_FIXABLE = frozenset({WRONG_TURN})
CRITIC_IMMUNE = frozenset({FELL_OVER, NO_GOOD_PLAN, TIMEOUT_ON_TRACK, UNREACHABLE_FAR,
                           GOAL_RADIUS_ARTIFACT, OFF_GRAPH, OSCILLATION})

# Candidate-pool verdicts (the X-vs-Z separator of Tier-1 #9).
POOL_GOOD_NOT_PICKED = "good_not_picked"     # critic's fault (ranking)
POOL_NO_GOOD = "no_good_candidate"           # proposal's fault (coverage)
POOL_GOOD_PICKED = "good_picked"             # selection was fine -> execution downstream
POOL_UNKNOWN = "unknown"                     # no candidate data available at the junction


@dataclass(frozen=True)
class ClassifierConfig:
    """Thresholds for the decision rules — all named, all in one place for the auditor.

    Defaults are conservative round numbers, not tuned to a result. Distances are
    BFS cells; antmaze-large is ~12x9 cells at scaling 4.0, so 1 cell ~= 4 world
    units ~= the policy's stride reach in a couple of steps.
    """
    goal_radius: float = 0.5          # GOAL_RADIUS_ARTIFACT: executed within this many WORLD
                                      #   units of the goal yet success=0. d4rl antmaze's reward
                                      #   radius IS 0.5, so this is a true termination/stride-hop
                                      #   miss — NOT the ~4-unit BFS-cell touch the old rule used
                                      #   (F1). Needs the executed xy; absent it, the rule is skipped.
    upright_fallen: float = 0.5       # torso up-axis z-component (R22) below this == toppled
    height_fallen: float = 0.3        # torso height (world z) below this == collapsed
    fall_window: int = 30             # FELL_OVER judges "still down" over the last this-many steps
    fall_frac: float = 0.5            # ...and requires the ant down for >= this fraction of them.
                                      #   A transient gait lean (R22 dips while turning) is NORMAL
                                      #   locomotion and must NOT count as a fall: the GPU run showed
                                      #   the old first-dip rule over-fired (11/14 FELL_OVER were
                                      #   oracle-fixable wrong turns).
    good_candidate_margin: float = 1.0  # a candidate >= this many cells closer than now == goalward/"good"
    wrong_turn_rise_frac: float = 0.5   # backslide after the min >= this fraction of the gained ground
    stuck_frac: float = 0.15          # closest approach within this fraction of start dist == never got going
    osc_reversals: int = 6            # >= this many progress-direction reversals == oscillation
    osc_net_frac: float = 0.25        # ...and net progress < this fraction of start == ping-ponging
    tail_window: int = 10             # number of tail steps used to judge "still progressing"
    tail_progress_cells: float = 1.0  # tail must close >= this many cells to count as on-track
    offgraph_frac_bad: float = 0.25   # >= this fraction off-graph == physically off the maze


# ── progress-curve features ─────────────────────────────────────────────────────

@dataclass
class ProgressFeatures:
    n_steps: int
    off_graph_frac: float
    start_dist: float
    min_dist: float
    argmin_step: int          # step index of the closest (best) approach
    end_dist: float
    net_progress: float       # start_dist - end_dist (positive == net closer)
    gained: float             # start_dist - min_dist (how much ground was gained at best)
    backslide: float          # end_dist - min_dist (how much was given back by the end)
    reversals: int            # progress-direction sign changes (oscillation signal)
    tail_closing: float       # cells closed over the last tail_window finite steps (>0 == still progressing)
    all_off_graph: bool


def _finite_pairs(dist: Sequence[float]) -> List[Tuple[int, float]]:
    """(step, value) for the finite (on-graph) samples, in order."""
    return [(i, d) for i, d in enumerate(dist) if math.isfinite(d)]


def progress_features(dist: Sequence[float], cfg: ClassifierConfig = ClassifierConfig()
                      ) -> ProgressFeatures:
    """Summarise a per-step BFS-distance-to-goal curve into shape features.

    `dist[t]` = geodesic cell distance from the executed state at step t to the
    goal; `inf` for an off-graph state (in-wall / unreachable). All shape features
    are computed on the finite samples; the off-graph fraction is tracked alongside.
    """
    n = len(dist)
    fin = _finite_pairs(dist)
    off_frac = 1.0 - (len(fin) / n if n else 0.0)
    if not fin:
        return ProgressFeatures(n_steps=n, off_graph_frac=off_frac, start_dist=INF,
                                min_dist=INF, argmin_step=-1, end_dist=INF,
                                net_progress=0.0, gained=0.0, backslide=0.0,
                                reversals=0, tail_closing=0.0, all_off_graph=True)
    start_dist = fin[0][1]
    end_dist = fin[-1][1]
    argmin_step, min_dist = min(fin, key=lambda p: p[1])
    # progress-direction reversals on the finite series (ignore exact ties)
    reversals, prev_sign = 0, 0
    for (_, a), (_, b) in zip(fin, fin[1:]):
        d = b - a
        sign = (d > 0) - (d < 0)
        if sign != 0:
            if prev_sign != 0 and sign != prev_sign:
                reversals += 1
            prev_sign = sign
    # tail behaviour: cells closed over the final stretch. Cap the window at half
    # the finite length so it is always a *genuine* tail — otherwise on a short
    # curve `tail_window` swallows the whole thing and this just re-measures net
    # progress (which would call a backslide "still closing").
    tail_n = min(cfg.tail_window, max(2, len(fin) // 2)) if len(fin) >= 2 else len(fin)
    tail = fin[-tail_n:]
    tail_closing = (tail[0][1] - tail[-1][1]) if len(tail) >= 2 else 0.0
    return ProgressFeatures(
        n_steps=n, off_graph_frac=off_frac, start_dist=start_dist, min_dist=min_dist,
        argmin_step=argmin_step, end_dist=end_dist,
        net_progress=start_dist - end_dist, gained=start_dist - min_dist,
        backslide=end_dist - min_dist, reversals=reversals,
        tail_closing=tail_closing, all_off_graph=False)


# ── pose collapse (execution failure) ───────────────────────────────────────────

def pose_collapse_step(upright: Sequence[float], height: Sequence[float],
                       cfg: ClassifierConfig = ClassifierConfig()) -> Optional[int]:
    """First step at which the torso is below the topple threshold (transient onset).

    A low-level primitive: `upright[t]` = world-z of the torso's local up-axis
    (1 = upright, <0 = flipped); `height[t]` = torso world z. Returns the FIRST dip,
    which may be a transient gait lean — use `sustained_collapse_onset` for the
    classifier's fall decision. Curves may be empty (non-antmaze) -> None.
    """
    n = min(len(upright), len(height))
    for t in range(n):
        if upright[t] < cfg.upright_fallen or height[t] < cfg.height_fallen:
            return t
    return None


def sustained_collapse_onset(upright: Sequence[float], height: Sequence[float],
                             cfg: ClassifierConfig = ClassifierConfig()) -> Optional[int]:
    """Onset of a SUSTAINED collapse (the ant ended on its back), or None.

    The classifier's fall signal. The ant must be down for >= `fall_frac` of the
    last `fall_window` steps — a transient lean that recovers is NOT a fall (the GPU
    run showed the first-dip rule mislabeled recoverable wrong turns as FELL_OVER).
    Returns the first step of the final contiguous down-run (the onset of staying
    down), or None if it recovered by the end.
    """
    n = min(len(upright), len(height))
    if n == 0:
        return None
    down = [(upright[t] < cfg.upright_fallen or height[t] < cfg.height_fallen)
            for t in range(n)]
    w = min(cfg.fall_window, n)
    if sum(down[-w:]) < cfg.fall_frac * w:
        return None                                   # recovered by the end -> not a fall
    onset = n
    for t in range(n - 1, -1, -1):                    # walk back over the final down-run
        if down[t]:
            onset = t
        else:
            break
    return onset


# ── candidate-pool quality (the ranking-vs-proposal separator) ──────────────────

def candidate_pool_quality(cand_dists: Sequence[float], chosen_dist: float,
                           cur_dist: float,
                           cfg: ClassifierConfig = ClassifierConfig()
                           ) -> Tuple[str, Dict[str, float]]:
    """At a decision step, did a goalward option exist, and was it picked?

    Inputs are oracle cell-distances-to-goal of each candidate's *endpoint*
    (`cand_dists`), of the *chosen* candidate's endpoint (`chosen_dist`), and of
    the current executed state (`cur_dist`). A candidate is "good" if its endpoint
    is at least `good_candidate_margin` cells closer to the goal than now.

    Returns one of POOL_NO_GOOD / POOL_GOOD_NOT_PICKED / POOL_GOOD_PICKED, which
    splits a wrong-turn into the critic's fault (a good option went unpicked),
    the proposal's fault (no good option existed), or execution (a good option
    *was* picked yet the rollout still failed downstream).
    """
    finite = [d for d in cand_dists if math.isfinite(d)]
    ev: Dict[str, float] = dict(cur_dist=cur_dist, chosen_dist=chosen_dist,
                                n_finite=float(len(finite)),
                                n_total=float(len(cand_dists)))
    if not finite or not math.isfinite(cur_dist):
        return POOL_NO_GOOD if finite == [] else POOL_UNKNOWN, ev
    best = min(finite)
    ev["best_dist"] = best
    good_threshold = cur_dist - cfg.good_candidate_margin
    good_exists = best <= good_threshold
    chosen_good = math.isfinite(chosen_dist) and chosen_dist <= good_threshold
    if not good_exists:
        return POOL_NO_GOOD, ev
    if not chosen_good:
        return POOL_GOOD_NOT_PICKED, ev
    return POOL_GOOD_PICKED, ev


# ── the classifier ──────────────────────────────────────────────────────────────

@dataclass
class FailureRecord:
    """Everything the classifier needs about one failed rollout (all plain data)."""
    dist: Sequence[float]                       # per-step BFS cell distance to goal
    upright: Sequence[float] = field(default_factory=list)
    height: Sequence[float] = field(default_factory=list)
    success: bool = False
    reach_step: Optional[int] = None
    is_far: bool = False                        # start dist in the global top quantile (set by analyzer)
    # min executed Euclidean (WORLD-unit) distance to the goal over the rollout, for
    # the goal-radius artifact test; None when the executed xy is unavailable.
    min_world_dist: Optional[float] = None
    # candidate pool at the failure junction (argmin step); optional
    junction_cand_dists: Optional[Sequence[float]] = None
    junction_chosen_dist: Optional[float] = None


def classify_failure(rec: FailureRecord, cfg: ClassifierConfig = ClassifierConfig()
                     ) -> Tuple[str, Dict[str, object]]:
    """Assign ONE primary failure mode, by a fixed priority (first match wins).

    Priority — numbered to match the CODE ORDER below. The junction candidate-pool
    verdict is consulted EARLY (recalibration): "a goalward candidate existed but the
    critic's pick led away" (`good_not_picked`) is the most direct ranking-failure
    signal and is exactly what the oracle-V ceiling exploits, so it must win over the
    pose / horizon guards that previously swallowed these cases.
      0. SUCCESS guard (shouldn't be called on a success).
      1. GOAL_RADIUS_ARTIFACT — executed within the true reward radius (world units)
         yet success=0 (measurement / stride-hop). Skipped if executed xy is absent.
      2. WRONG_TURN — junction pool `good_not_picked` (the critic's chosen plan was
         not goalward though a goalward candidate existed). Critic-fixable.
      3. FELL_OVER — SUSTAINED collapse (ant ended on its back), no goalward option
         going unpicked (execution). Transient gait leans do not count.
      4. OFF_GRAPH — mostly off the maze graph without a clean topple (clip).
      5. TIMEOUT_ON_TRACK / UNREACHABLE_FAR — still clearly progressing at the horizon
         (the chosen plans were goalward, just out of time). is_far is unreliable when
         every goal shares a corner (antmaze-large-diverse) — both are immune anyway.
      6. OSCILLATION — many reversals, little net progress (value-side already caught
         at 2, so this residual is policy/proposal ping-ponging; immune).
      7. NO_GOOD_PLAN / FELL_OVER — backslid/never-advanced with no goalward candidate
         (proposal) or a goalward one that WAS picked yet failed (execution).
      8. WRONG_TURN (degraded) — backslid/never-advanced with the pool unavailable.
      9. TIMEOUT / UNCLASSIFIED — guards.

    Returns (mode, evidence). Evidence carries the features + pool verdict so the
    analyzer can print why, and so a miscalibrated threshold is debuggable.
    """
    f = progress_features(rec.dist, cfg)
    ev: Dict[str, object] = dict(features=f, min_world_dist=rec.min_world_dist)

    if rec.success:
        return "SUCCESS", ev

    # 1. measurement artifact: within the TRUE reward radius (world units) yet
    #    success=0 — a termination / 25-step-stride-hop miss, not the coarse BFS-cell
    #    touch (F1). Without executed xy we SKIP rather than mislabel a real failure.
    if rec.min_world_dist is not None and rec.min_world_dist <= cfg.goal_radius:
        ev["why"] = (f"executed within {cfg.goal_radius} world units of goal "
                     f"(min={rec.min_world_dist:.2f}) but success=0")
        return GOAL_RADIUS_ARTIFACT, ev

    # Junction pool verdict (closest-approach step) — computed up front so the
    # ranking signal can pre-empt the pose/horizon guards (recalibration).
    pool_verdict, pool_ev = POOL_UNKNOWN, {}
    if rec.junction_cand_dists is not None and rec.junction_chosen_dist is not None \
            and f.argmin_step >= 0 and math.isfinite(f.min_dist):
        pool_verdict, pool_ev = candidate_pool_quality(
            rec.junction_cand_dists, rec.junction_chosen_dist, f.min_dist, cfg)
    ev["pool_verdict"], ev["pool_evidence"] = pool_verdict, pool_ev

    # 2. ranking failure (critic-fixable): a goalward candidate existed at the
    #    junction but the chosen plan was NOT goalward. This is the oracle-V signal
    #    and wins over fall/horizon (a fall reached via a bad pick is still a bad
    #    pick; the oracle ceiling confirms these are recoverable by selection alone).
    if pool_verdict == POOL_GOOD_NOT_PICKED:
        ev["why"] = "goalward candidate went unpicked at the junction"
        return WRONG_TURN, ev

    # 3. execution: a SUSTAINED collapse (ant ended on its back). Transient gait leans
    #    do not qualify (recalibration); and a fall with a better pick available was
    #    already routed to WRONG_TURN above.
    collapse = sustained_collapse_onset(rec.upright, rec.height, cfg)
    ev["collapse_step"] = collapse
    if collapse is not None:
        ev["why"] = f"sustained collapse from step {collapse} (down at episode end)"
        return FELL_OVER, ev

    # 4. physically off the maze graph without a clean pose collapse (e.g. clipped)
    if f.all_off_graph or f.off_graph_frac >= cfg.offgraph_frac_bad:
        ev["why"] = f"off-graph frac {f.off_graph_frac:.2f}"
        return OFF_GRAPH, ev

    backslid = f.backslide >= cfg.wrong_turn_rise_frac * f.gained and f.gained > 0
    never_advanced = f.min_dist >= f.start_dist * (1.0 - cfg.stuck_frac)
    on_track = (f.tail_closing >= cfg.tail_progress_cells and f.net_progress > 0)
    oscillating = (f.reversals >= cfg.osc_reversals
                   and f.net_progress < cfg.osc_net_frac * max(f.start_dist, 1.0))

    # 5. still clearly progressing at the tail with goalward picks -> horizon-limited
    if on_track and not backslid:
        mode = UNREACHABLE_FAR if rec.is_far else TIMEOUT_ON_TRACK
        ev["why"] = (f"still closing at tail ({f.tail_closing:.1f} cells) "
                     f"net_progress={f.net_progress:.1f}")
        return mode, ev

    # 6. oscillation residual (value-side already taken at 2) -> immune ping-ponging
    if oscillating:
        ev["why"] = f"{f.reversals} reversals, net_progress={f.net_progress:.1f}"
        return OSCILLATION, ev

    # 7/8. wrong-turn family: advanced-then-backslid, or never advanced at all
    if backslid or never_advanced:
        if pool_verdict == POOL_NO_GOOD:
            ev["why"] = "no goalward candidate at the junction"
            return NO_GOOD_PLAN, ev
        if pool_verdict == POOL_GOOD_PICKED:
            # a good option WAS picked yet it failed -> downstream execution, not ranking
            ev["why"] = "good candidate picked at the junction but failed downstream"
            return FELL_OVER, ev
        # pool unavailable -> degraded best-guess of ranking, evidence kept honest (F5)
        ev["why"] = ("advanced then backslid; pool unavailable" if backslid
                     else "never advanced toward the goal; pool unavailable")
        return WRONG_TURN, ev

    # 9. fell through every rule — slow/quiet timeout, treat as horizon-limited
    if f.net_progress > 0:
        mode = UNREACHABLE_FAR if rec.is_far else TIMEOUT_ON_TRACK
        ev["why"] = f"net_progress={f.net_progress:.1f}, no decisive shape"
        return mode, ev
    ev["why"] = "no rule matched"
    return UNCLASSIFIED, ev


def tally(modes: Sequence[str]) -> List[Tuple[str, int, float]]:
    """(mode, count, pct) sorted by count desc — the Tier-1 output table rows."""
    n = len(modes)
    counts: Dict[str, int] = {}
    for m in modes:
        counts[m] = counts.get(m, 0) + 1
    rows = [(m, c, 100.0 * c / n if n else 0.0) for m, c in counts.items()]
    rows.sort(key=lambda r: -r[1])
    return rows
