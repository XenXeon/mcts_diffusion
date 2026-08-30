"""mcts/value_scale.py

The pipeline's value scale, made explicit and shared (plan v5.1 §3a, item R5.7a).

The DV dataset maps raw per-state values (negative steps-to-terminus at discount 1.0)
to [-1, 1] by a global min-max + center mapping. With vmax_raw = 0 (a state at a
terminus) and vmin_raw = -D (the deepest steps-to-go in the dataset), the affine is

    val(d) = 1 - 2*d/D          (d = steps-to-go;  val(0) = 1.0 exactly)
    steps(v) = (1 - v) * D / 2  (inverse)

Relabelled V(s, g) targets MUST pass through this identical affine — never refit
min-max on the relabelled offset distribution — so that:
  * target(t' = t) = 1.0 exactly = the point-to-segment terminal-check value, and
  * V(s), V(s, g), and terminal children all sit on one scale inside the same
    MAX backup.

D is recovered from the dataset as the maximum terminus index over paths (for
learn_policy=False antmaze data every stored path reaches a terminus, and the raw
value of the path start is -terminus_index). `assert_consistent_with_dataset`
verifies the recovered affine against the dataset's own seq_val.

Torch/numpy-free on purpose: unit-tested locally; the trainer and diagnostics
import it on the GPU box.
"""
from __future__ import annotations

from typing import Sequence


class StepScale:
    """Affine between steps-to-go and the pipeline's [-1, 1] value scale."""

    def __init__(self, D: int) -> None:
        if D <= 0:
            raise ValueError(f"D must be a positive step count, got {D}")
        self.D = int(D)

    def val(self, d: float) -> float:
        """steps-to-go -> value in [-1, 1]; offsets beyond D clip at -1."""
        if d < 0:
            raise ValueError(f"steps-to-go must be >= 0, got {d}")
        return max(-1.0, 1.0 - 2.0 * d / self.D)

    def val_array(self, d):
        """Vectorised `val` for a numpy array of steps-to-go (the training hot
        path). Numpy-import-free: uses ndarray arithmetic + `.clip` by duck
        typing, so this stays the single definition of the affine (C2) without
        forcing numpy into this module."""
        return (1.0 - 2.0 * d / self.D).clip(-1.0)

    def steps(self, v):
        """value -> steps-to-go (inverse of val on the unclipped range).
        Pure arithmetic — already array-safe, so diagnostics call it directly."""
        return (1.0 - v) * self.D / 2.0

    @classmethod
    def from_terminus_indices(cls, terminus_indices: Sequence[int]) -> "StepScale":
        """D = max steps-to-go in the dataset = max terminus index over paths."""
        if len(terminus_indices) == 0:
            raise ValueError("no terminus indices supplied")
        return cls(max(int(t) for t in terminus_indices))

    def assert_consistent_with_dataset(self, seq_val_start: float,
                                       terminus_index: int,
                                       tol: float = 1e-4) -> None:
        """Check the recovered affine reproduces the dataset's own normalisation.

        For a path with terminus index T, the dataset's normalised value at the
        path start must equal val(T). A mismatch means D was recovered wrongly
        (or the dataset's target config changed) — fail loudly, never silently
        train on a shifted scale.
        """
        expect = self.val(terminus_index)
        if abs(expect - float(seq_val_start)) > tol:
            raise AssertionError(
                f"value-scale mismatch: dataset seq_val at path start = "
                f"{seq_val_start:.6f}, recovered affine gives val({terminus_index}) "
                f"= {expect:.6f} with D={self.D} — do NOT train on this scale")
