# Results subsection DRAFT (fix 3) — value posedness does not yield a reliable DV-tree gain

Standalone draft (2026-07-13, revised after the generalization runs). This
**confirms** the "value lever is exhausted on the DV backbone" claim rather than
overturning it, and it is written to slot into the results chapter as supporting
evidence for §2's negative result — *not* as a new positive result. (An earlier
version of this file drafted it as a headline win; the medium/umaze runs killed
that framing. See the integration note at the end.)

---

## 6c. Testing the best-posed value: a goal-conditioned V(s, g) across the maze2d family

§2 established that per-state V(s) ties MCSS on navigation because its target is
ill-posed (a single state does not determine its return under a random goal).
That analysis tested only ill-posed values. To close the value lever properly we
tested the value the pivot's own analysis identified as *well-posed*: a
goal-conditioned V(s, g) whose target — normalized time to reach a spatially
specified goal — is deterministic given (s, g). We enabled it on maze2d (adding
the terminus labels the relabeler needs; the goal is the observation's xy, and
the target is time-to-reach), trained it to strong held-out fit on all three
maze2d sizes (val_corr 0.88 / 0.80 / 0.62 on large / medium / umaze), and used
its pessimistic ensemble-minimum as the DV tree's node value. If any value could
make the DV tree beat MCSS, this is it.

**It does not — reliably.** Start-matched, compute-matched (vs wide MCSS, k256):

| env | V(s, g) val_corr | V(s, g)-pess tree − compute-matched MCSS |
|---|---|---|
| maze2d-large | 0.88 | **+3.98** (seed-t = 2.83, 10 seeds) — win |
| maze2d-medium | 0.80 | **−14.82** (seed-t = −2.69, 5 seeds) — loss |
| maze2d-umaze | 0.62 | **−4.90** (seed-t = −3.06, 5 seeds) — loss |

The same well-posed pessimistic value, driving the same tree, **wins on one
maze size and loses significantly on the other two** — and the losses are not a
value-quality artifact: maze2d-medium's value is strong (val_corr 0.80, close to
large's 0.88) yet its tree is the worst of the three. The large win is therefore
an outlier, most plausibly a property of large's long corridors (where a
goal-distance value's "head straight for the goal" preference is executable) that
does not hold on the tighter medium/umaze layouts (where it over-commits to
aggressive plans the inverse-dynamics policy cannot follow — the same
imagination/execution gap seen in §4). Averaged over the family, value posedness
buys no reliable gain.

**This confirms, at its well-posed limit, that the value lever is exhausted on
the DV backbone.** The negative result of §2 ("a better *learned* value does not
let the DV tree beat MCSS") is not an artifact of testing only ill-posed values:
the best-posed value available — goal-conditioned, pessimistic, well-fit — also
fails to deliver a reliable win. The single route to a robust tree gain remains
the one §7 identifies: a faithful-conditioning backbone (Diffusion Forcing),
whose win replicates across two environments and backbones. The RQ0 answer is
unchanged — search does not beat MCSS on the frozen DV planner by improving the
node value, however well-posed.

**Scope.** V(s, g) is a spatial time-to-goal value and applies only to the
navigation environments; it is undefined on kitchen (no spatial goal — the
observation's leading dimensions are joint angles and the goal is a subtask set).
Antmaze is locomotion-capped, so it is not an informative test of a
selection-value's effect and is omitted.

---

## Integration note

- Slot after §2 (the negative-result section) or into the discussion's
  "value lever" paragraph — as **support**, not a new claim.
- Do **not** headline the maze2d-large number; report the three-size picture and
  the outlier reading. Reporting only the large win would be the credibility
  problem the drafts were being fixed to avoid.
- The methodology gains a one-line note: V(s, g) enabled on maze2d via the
  `seq_tml` terminus derivation (the maze2d dataset segments paths at a
  goal-reach, giving a genuine spatial terminus).
