# maze2d-large DV-backbone numbers — start-matched correction (2026-07-28)

*Single source of truth for the maze2d DV rows. Supersedes every cross-start figure
previously quoted in the chapter drafts. Every number below was recomputed from the
per-rollout `dv_norm` vectors in `results/`, with the `starts` arrays asserted
identical between the compared arms (assertion printed in the audit; no comparison
below is cross-start).*

---

## 1. What was actually wrong

The runbook recorded "204.9 is a cross-start artifact." That is **half right, and the
half that is wrong matters.**

`results/maze2d_large_critic_tree.json` is a `--method both` run at seed 0 with
`n_envs=25 × n_episodes=6 = 150` rollouts. Its MCSS arm scores **204.89** and its
`starts` array is **identical to its own tree arm's**. So 204.9 is a perfectly valid,
internally-paired DV-MCSS k50 measurement — *on its own 150-start set*.

The error was never the 204.9 measurement. It was **using 204.9 as the comparator for
tree arms measured on a different start set**. The n=25 seed-0 tree arms
(`m2l_tree_r50_s0`, `m2l_tree_r50_m3_s0`, `m2l_tree_criticr50_inpaint`,
`m2l_tree_planv_s0`) all run on exactly the **first 25 starts** of that 150-start set
(verified: `ref25 == ref150[:25]`), whose MCSS mean is **200.61**, not 204.89. Comparing
a 25-start tree arm against a 150-start baseline mean imported a **−4.3 point offset**
into every delta.

The seeds 0–9 pool (`maze2d_large_mcss_k50_s0..s9`, 25 envs × 1 episode × 10 seeds,
n=250) is a **third** start set, mean **199.36**. It is the right baseline for the
10-seed tree arms only.

**Rule going forward:** never quote a delta between arms from different files without
asserting `np.allclose(a['starts'], b['starts'])` first.

## 2. Corrected baselines (start-matched)

| baseline | score | n | start set |
|---|---|---|---|
| DV-MCSS k50, seeds 0–9 | **199.36** | 250 | seeds 0–9, 1 ep |
| DV-MCSS k256, seeds 0–9 | **201.19** | 250 | seeds 0–9, 1 ep (same starts as k50) |
| DV-MCSS k50, 150-start ref set | 204.89 | 150 | seed 0, 6 ep — *valid, but only its own arms* |
| DV-MCSS k50, first 25 of ref set | 200.61 | 25 | comparator for the n=25 seed-0 tree arms |

Width control, **start-matched**: k50 199.36 → k256 201.19 = **+1.83 for 5× the
samples**. Width saturates. (The previously quoted `k256 = 206.0` was cross-start and
is withdrawn.)

## 3. Corrected tree deltas (all start-matched, starts asserted identical)

| arm | score | vs matched MCSS k50 | statistic | n | seeds |
|---|---|---|---|---|---|
| naive critic tree, original cfg | 194.3 | **−5.05** | seed-t −10.57, roll-t −7.05 | 250 | 0–9 |
| tree MAX backup, glue | 197.7 | **−4.29** | seed-t −4.75, roll-t −3.67 | 75 | 0–2 |
| tree top-3 backup, glue | 202.2 | **+0.25** | seed-t +0.34 (n.s.) — **parity** | 75 | 0–2 |
| tree plan-value V̂(s) | 202.5 | **+0.52** | seed-t +0.30 (n.s.) — **parity** | 75 | 0–2 |
| tree inpaint, MAX | 182.1 | **−18.53** | roll-t −3.02 | 25 | 0 |
| **MAX → top-3 (winner's curse fix)** | — | **+4.54** | roll-t +5.00 | 75 | 0–2 |

The winner's-curse figure is **unchanged** by this correction, and always was valid:
it is a difference between two arms run on *identical* starts, so the baseline cancels.
The previously quoted +4.5 is confirmed at +4.54.

## 4. Old → new mapping (for the edit pass)

| quoted in drafts | replace with | conclusion |
|---|---|---|
| DV-MCSS k50 = 204.9 | **199.4** (10 seeds) | unchanged role |
| DV-MCSS k256 = 206.0 | **201.2** | unchanged role |
| naive tree −6.8 | **−5.05, seed-t −10.57, 10 seeds** | *stronger* than before |
| MAX-backup tree −5.7 | **−4.29, seed-t −4.75, 3 seeds** | unchanged in kind |
| top-3 tree −2.6 | **+0.25, n.s. — exact parity** | **sign flips; "parity" now literal** |
| inpaint −22.8 | **−18.53, roll-t −3.02** | unchanged in kind |
| top-3 fix +4.5, p<1e-4 | **+4.54, roll-t +5.00** | confirmed |
| plan-value tree ≈ parity | **+0.52, n.s.** | confirmed |

## 5. What the correction does and does not change

**Unchanged (all qualitative conclusions survive, most are better supported):**
- Naive tree search loses on the full-sequence DV backbone — now at 10 seeds with
  seed-t −10.6 and negative on *every* seed, far stronger than the original
  single-seed −6.8.
- Every repair buys parity and never a win: top-3 +0.25 (n.s.), plan-value +0.52 (n.s.).
- The winner's curse and its tempered-backup fix: +4.54, roll-t 5.00.
- Prefix-inpainting is catastrophic: −18.5, and remains by far the worst arm.
- Width saturates, so the deficit is not sampling volume.
- The expansion-fidelity triptych keeps its monotone ordering:
  **glue −4.3 < inpaint −18.5 < DF +9.0** (each vs its own start-matched flat baseline).

**Changed in wording:**
- "the DV tree reaches 202.3 (−2.6): parity … never a win" becomes *literal* parity
  (+0.25, n.s.). The claim is now more accurate, not less: the fixed DV tree matches
  its flat baseline exactly, at ~6× the compute.

**Retired:**
- The owed maze2d compute-matched control (k≈290). The start-matched k256 arm at
  seeds 0–9 **is** that control: it draws 256 flat samples against the tree's ~290 and
  gains only +1.8 over k50, so the DV tree's parity is not a sampling-volume artifact.
  Limitation "the maze2d k≈290 control never ran" should be struck wherever it appears.

## 6. Provenance

Audit performed 2026-07-28 directly against `results/*.json`; every comparison asserted
`starts` equality before differencing. Note `scripts/analyze_maze2d_values.py` pools
mixed configurations (MAX / top-3 / stitched) into its critic and V(s) rows — its
−3.88 critic figure is a config-mixed average and should **not** be quoted. The
per-config figures in §3 above supersede it.
