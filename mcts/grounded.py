"""mcts/grounded.py

Grounded (non-learned) subtask-completion checker for FrankaKitchen
(kitchen-mixed-v0 and siblings).

On kitchen-mixed-v0 every LEARNED value in this stack — the DV trajectory
critic, V(s), V(s,g), the noise-aware critic — is trained on labels derived
from the demonstration dataset, and NO demonstration in kitchen-mixed ever
solves all 4 subtasks (verified by census). A value trained on capped labels
cannot express "this plan finishes the 4th subtask" versus "this plan
finishes only the same 3 subtasks a different way" — both score identically
at the label ceiling, because the training data never showed the learner the
difference.

This module computes subtask completion the way the ENVIRONMENT itself does:
straight from the env's own OBS_ELEMENT_INDICES / OBS_ELEMENT_GOALS /
BONUS_THRESH task definitions (cleandiffuser/env/kitchen/base.py,
KitchenBase._get_reward_n_score), not from a learned function of them. It is
therefore exempt from the label cap — the ONE evaluator in this repo that can
look at a window and correctly say "the 4th subtask got solved here", whether
or not any training trajectory ever did the same.

Two use cases:
  (a) open-loop diagnostic (scripts/check_grounded_pool.py): can the frozen
      DF planner even IMAGINE a state beyond the dataset's demonstrated
      ceiling?
  (b) an optional tree node value / MCSS reranker (mcts/mcts_loop.py,
      value_mode="grounded" / --grounded-mcss).

Kept in two layers, same convention as mcts/df_schedule.py: a torch-free
numpy core (cumulative_solved_count — importable on this repo's torch-free
Windows dev box and unit-tested there, tests/test_grounded.py) underneath a
torch wrapper (KitchenGroundedChecker) used by the closed-loop code.
"""
from __future__ import annotations

import sys

import numpy as np


def cumulative_solved_count(windows: np.ndarray, elem_indices: list[np.ndarray],
                            elem_goals: list[np.ndarray], thresh: float) -> np.ndarray:
    """(n, T, D) RAW (unnormalized) obs windows -> (n,) float64 grounded subtask count.

    For task element e (obs dims elem_indices[e], goal vector elem_goals[e]),
    element e is "solved at step t" iff the L2 norm of
    (windows[:, t, elem_indices[e]] - elem_goals[e]) is STRICTLY less than
    thresh — exactly KitchenBase._get_reward_n_score's own bonus condition
    (cleandiffuser/env/kitchen/base.py), so this reproduces the env's own
    accounting rather than approximating it.

    The returned count for row i is the size of the UNION, over ALL elements
    and ALL steps t = 0..T-1 (INCLUDING row 0, the window's current/starting
    state), of the elements solved at that step:

        count[i] = |{ e : exists t in [0, T) s.t. solved(i, e, t) }|

    Row 0 is included deliberately: it is the state the plan starts FROM, and
    the env's own bookkeeping is monotone (KitchenBase.REMOVE_TASKS_WHEN_COMPLETE
    — a subtask solved earlier stays solved), so an already-completed subtask
    must count toward the plan's total exactly like the environment counts
    it. Concretely: a window planned FROM a state with 3 subtasks already
    done, whose CONTINUATION reaches the 4th element's goal at some later
    step, scores 4 — that is precisely the preference no learned value in
    this stack can express, because no training label was ever 4.

    Vectorized over n and t (only the small loop over ~4 task elements is a
    Python loop — deliberately: elem_indices[e] can have a different WIDTH
    per element, e.g. kitchen's "kettle" is 7-dim and "microwave" is 1-dim,
    so the elements cannot be stacked into one array without padding).

    Raises ValueError on malformed shapes or thresh <= 0.
    """
    windows = np.asarray(windows)
    if windows.ndim != 3:
        raise ValueError(f"windows must be (n, T, D), got shape {windows.shape}")
    n, T, D = windows.shape
    if thresh <= 0:
        raise ValueError(f"thresh must be > 0, got {thresh}")
    if len(elem_indices) != len(elem_goals):
        raise ValueError(f"elem_indices and elem_goals must have the same length, "
                         f"got {len(elem_indices)} vs {len(elem_goals)}")
    if len(elem_indices) == 0:
        raise ValueError("elem_indices/elem_goals must be non-empty")
    total = np.zeros(n, dtype=np.float64)
    for idx, goal in zip(elem_indices, elem_goals):
        idx = np.asarray(idx)
        goal = np.asarray(goal, dtype=np.float64)
        if idx.ndim != 1 or goal.ndim != 1 or idx.shape[0] != goal.shape[0]:
            raise ValueError(f"element index/goal shape mismatch: idx {idx.shape} "
                             f"vs goal {goal.shape} (both must be 1-D, equal length)")
        if idx.size == 0:
            raise ValueError("an element's index array is empty")
        if int(idx.max()) >= D or int(idx.min()) < 0:
            raise ValueError(f"element index {idx.tolist()} out of bounds for "
                             f"obs dim D={D}")
        diff = windows[:, :, idx].astype(np.float64) - goal[None, None, :]  # (n, T, w)
        dist = np.linalg.norm(diff, axis=-1)                                # (n, T)
        solved_any_t = (dist < thresh).any(axis=1)                          # (n,)
        total += solved_any_t.astype(np.float64)
    return total


# Torch is optional at import time: the numpy core above is what
# tests/test_grounded.py imports, and it must stay importable on a torch-free
# box (this repo's local Windows dev machine has no torch installed). Only
# KitchenGroundedChecker below needs torch, and only when its methods are
# actually called — `import mcts.grounded` itself must never require torch.
try:
    import torch
except ImportError:                       # pragma: no cover - the torch-free
    torch = None                          # test box itself exercises this path


def _lookup_task_attr(e, name: str):
    """Find a kitchen task-definition constant: on the env instance first, then
    in the defining MODULE of each class in the env's MRO.

    Both kitchen implementations this repo can load (d4rl's kitchen_envs and
    cleandiffuser/env/kitchen/base.py) keep TASK_ELEMENTS as a CLASS attribute
    but OBS_ELEMENT_INDICES / OBS_ELEMENT_GOALS / BONUS_THRESH as MODULE-level
    constants next to the class — hasattr(env, ...) alone finds the first and
    misses the other three (the exact crash the first deployment hit on
    KitchenMicrowaveKettleBottomBurnerLightV0). Returns None if not found
    anywhere; the caller raises with the full search trail.
    """
    if hasattr(e, name):
        return getattr(e, name)
    for klass in type(e).__mro__:
        mod = sys.modules.get(getattr(klass, "__module__", None))
        if mod is not None and hasattr(mod, name):
            return getattr(mod, name)
    return None


class KitchenGroundedChecker:
    """Torch wrapper: reads task definitions off a LIVE kitchen env, then
    scores (normalized) torch windows via cumulative_solved_count.

    Deliberately reads NO hardcoded element/goal/threshold constants —
    copying OBS_ELEMENT_INDICES/GOALS by hand into this file would silently
    drift from whatever env variant is actually loaded (KitchenAllV0 has 7
    task elements; the 4-element variant used at training time has 4) and
    produce a confidently-wrong score with no error. A missing/renamed
    attribute instead raises immediately in from_env — a loud failure at
    setup, not a silent corrupter of every downstream result.
    """

    def __init__(self, elem_indices: list[np.ndarray], elem_goals: list[np.ndarray],
                thresh: float, mean: np.ndarray, std: np.ndarray) -> None:
        self.elem_indices = elem_indices
        self.elem_goals = elem_goals
        self.thresh = float(thresh)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(-1)
        self.std = np.asarray(std, dtype=np.float32).reshape(-1)
        self._max_idx = max(int(np.asarray(idx).max()) for idx in elem_indices)

    @classmethod
    def from_env(cls, env, normalizer) -> "KitchenGroundedChecker":
        """Build from a LIVE env's task definitions + the dataset's GaussianNormalizer.

        env: the gym env (or its .unwrapped) that exposes TASK_ELEMENTS /
        OBS_ELEMENT_INDICES / OBS_ELEMENT_GOALS / BONUS_THRESH as class or
        instance attributes (cleandiffuser/env/kitchen/base.py KitchenBase and
        d4rl's own kitchen env share this attribute contract). Any missing
        attribute means this is not a kitchen env -> ValueError, loud and
        immediate, rather than a checker that silently scores nonsense.

        normalizer: the dataset's GaussianNormalizer (normalize = (x-mean)/std,
        so unnormalize = x*std+mean); its .mean/.std are stored as float32
        numpy for count()'s unnormalize step. A normalizer without .mean/.std
        (e.g. a different Normalizer subclass) is refused for the same reason.
        """
        e = getattr(env, "unwrapped", env)
        required = ("TASK_ELEMENTS", "OBS_ELEMENT_INDICES", "OBS_ELEMENT_GOALS",
                   "BONUS_THRESH")
        found = {a: _lookup_task_attr(e, a) for a in required}
        missing = [a for a, v in found.items() if v is None]
        if missing:
            raise ValueError(
                f"env (unwrapped type {type(e).__name__}) is missing kitchen "
                f"task-definition attribute(s) {missing} — searched the env "
                f"instance and the defining module of every class in its MRO "
                f"(d4rl/cleandiffuser keep OBS_ELEMENT_* and BONUS_THRESH at "
                f"module level). from_env only works on a FrankaKitchen env; "
                f"a silently-wrong goal set would corrupt every grounded score "
                f"downstream")
        obs_idx, obs_goals = found["OBS_ELEMENT_INDICES"], found["OBS_ELEMENT_GOALS"]
        task_elements = list(found["TASK_ELEMENTS"])
        if not task_elements:
            raise ValueError("env.TASK_ELEMENTS is empty — no kitchen subtasks defined")
        elem_indices, elem_goals = [], []
        for name in task_elements:
            if name not in obs_idx or name not in obs_goals:
                raise ValueError(
                    f"TASK_ELEMENTS contains {name!r} but OBS_ELEMENT_INDICES/"
                    f"OBS_ELEMENT_GOALS has no entry for it")
            elem_indices.append(np.asarray(obs_idx[name]))
            elem_goals.append(np.asarray(obs_goals[name], dtype=np.float64))
        thresh = float(found["BONUS_THRESH"])
        if thresh <= 0:
            raise ValueError(f"env.BONUS_THRESH must be > 0, got {thresh}")
        if not hasattr(normalizer, "mean") or not hasattr(normalizer, "std"):
            raise ValueError(
                f"normalizer lacks .mean/.std — from_env needs a GaussianNormalizer "
                f"(unnormalize = x*std + mean); got {type(normalizer).__name__}")
        return cls(elem_indices, elem_goals, thresh,
                   np.asarray(normalizer.mean), np.asarray(normalizer.std))

    def _check_dim(self, D: int) -> None:
        if D != self.mean.shape[0]:
            raise ValueError(f"input obs dim {D} != normalizer dim {self.mean.shape[0]}")
        if self._max_idx >= D:
            raise ValueError(f"max task-element obs index {self._max_idx} >= obs "
                             f"dim {D} (env/normalizer mismatch)")

    def count(self, x_norm: "torch.Tensor") -> "torch.Tensor":
        """(n, T, D) NORMALIZED windows -> (n,) float32 grounded solved-count.

        Unnormalizes (x*std+mean) before scoring: OBS_ELEMENT_GOALS are RAW
        physical targets (joint angles / hinge positions), meaningless on
        GaussianNormalizer-standardized dims. Asserts the obs dim matches the
        normalizer (and that every task-element index is in bounds) on every
        call — cheap, and a silent shape mismatch here would silently
        corrupt the score rather than error.
        """
        if torch is None:
            raise RuntimeError(
                "KitchenGroundedChecker.count needs torch, which is not "
                "installed in this environment")
        x_np = x_norm.detach().cpu().numpy()
        D = x_np.shape[-1]
        self._check_dim(D)
        raw = (x_np.astype(np.float64) * self.std[None, None, :].astype(np.float64)
              + self.mean[None, None, :].astype(np.float64))
        counts = cumulative_solved_count(raw, self.elem_indices, self.elem_goals,
                                         self.thresh)
        return torch.as_tensor(counts, dtype=torch.float32, device=x_norm.device)

    def score(self, x_norm: "torch.Tensor") -> "torch.Tensor":
        """(n, T, D) -> (n,) float32 in [-1, 1]: count/2 - 1.

        Maps the 0..4 grounded count onto the SAME [-1, 1] scale every other
        node value in this repo assumes (the learned critics' own training
        target; the junction-filter sentinel -10 in mcts_loop.py is defined
        relative to this range too) so it can be blended with / compared
        against critic scores without a separate rescale.
        """
        return self.count(x_norm) / 2.0 - 1.0
