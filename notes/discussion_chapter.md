# Discussion and Conclusion (dissertation chapter draft)

Draft 2026-07-12. Companion to `intro_chapter.md`, `related_work_chapter.md`,
`methodology_report.md`, `results_chapter.md`. This chapter interprets the
results as a whole, states the practical consequences, is candid about what
the study does and does not establish, and lays out the work that would move
the frontier — including the one intervention the boundary result points at.

---

## 5.1 What the experiments add up to

The project began as an engineering attempt — replace a diffusion planner's
flat best-of-K inference with tree search and win — and became a measurement
of *when that substitution can pay*. The answer is a three-clause law, and
each clause is carried by its own controlled experiment rather than by an
aggregate benchmark delta:

1. **Faithful expansion is the precondition.** Tree search over a
   full-sequence diffusion planner is harmful (−5.05 on maze2d, negative on
   every one of ten seeds), not because search is wrong but because the planner
   cannot natively condition a continuation on a search prefix; every workaround
   injects an artifact the evaluator mis-scores and the backup then exploits.
   The expansion-fidelity triptych — seam-glue (−4.3) worse than nothing,
   replacement inpainting catastrophic (−18.5), exact per-token conditioning
   positive (+9 to +10.5) — maps monotonically onto tree performance, and the
   winner's-curse signature that plagues the full-sequence backbone (MAX backup
   +4.54 below tempered backup, roll-t 5.00) *vanishes* under exact conditioning
   (MAX ≈ top-3). Search's failure was never search; it was lying node
   evaluations.

2. **The gain is the headroom the flat baseline leaves.** Across backbones of
   decreasing flat quality (DV → DF 183.4 → shortcut 148.3) the tree's gain grows
   monotonically (−2.2 → +9.0 → +37.2) while its *landing point* barely moves
   (202 → 192 → 185). (The DV figure is start-matched against the
   compute-matched k256 baseline of 201.2; against root width the same tree ties.
   See methodology §7.5 †.) The same shape
   reappears within one environment as generation-side guidance strengthens the
   flat pool. Search
   is a partial equaliser: it recovers most of what a weaker sampler discards
   and pays essentially nothing above a strong one. This reframes "does search
   help diffusion planners?" as a question with a quantitative answer that
   depends entirely on the baseline it is measured against — and explains why
   the strongest published planner (DV, at its critic's selection ceiling) is
   flat and guidance-free by design, while concurrent positive results (MCTD)
   are reported on long-horizon benchmarks whose flat baselines are weak.

3. **No learned-value method passes the dataset's demonstration ceiling.** On
   kitchen-mixed, whose 613 demonstrations never once solve all four subtasks,
   an 850+-rollout census spanning four method families — flat selection, tree
   search, and classifier guidance at both trajectory-level (the DV pipeline's
   own CG, 74.75) and token-level noise resolution — contains zero four-task
   completions, and the strongest baseline sits *exactly* on the ceiling
   (75.0 = three-of-four in every rollout). Inference-time structure improves
   reliability up to the best behaviour the data teaches; it does not invent
   behaviour the data never showed.

## 5.2 Two recurring dissociations, and why they matter

Beyond the headline law, two findings recurred independently and, taken
together, form the chapter's most transferable lesson.

**Imagination is not execution.** Twice, a method that improved what a value
model *scores* degraded what the policy can *do*. The shortcut-forcing
backbone produced plans the DV critic scored *above* the 52-step planner's
(open-loop gen 0.055 vs −0.003) yet executed worst (148.3), because its
few-step sampling left a physical hop-tail the inverse-dynamics policy could
not follow. The grounded evaluator — the one value in the stack exempt from
the label cap — ranked plans by *imagined* subtask touches and scored *below*
the learned critic in closed loop (flat 55.0 vs 60.0, tree 63.0 vs 70.0),
because it rewarded the transient near-goal grazes its union scoring admitted,
diverting the executed step toward completions that never cashed. The common
cause is that a value model optimised for one purpose (matching returns,
counting touched goals) is not automatically calibrated for another
(selecting executable plans). The DV critic's very conservatism — it scores
shape-realism, not optimistic achievement, which is *why* it capped at 75 —
turns out to be load-bearing for control. For anyone building
search-over-generation systems this is the practical warning: evaluate the
evaluator on downstream execution, not on its own held-out accuracy.

**Search amplifies evaluator error, it does not launder it.** The winner's
curse is usually stated as a generic maximisation bias; here it was localised
to a mechanism (off-manifold expansion seams the critic mis-scores) and shown
to be *fixable at the source* (exact conditioning) rather than only at the
backup (tempering). A corollary that held throughout: search never rescued a
mis-calibrated value. The tree added its structural +8 over the grounded flat
baseline just as it did over the critic flat baseline — composition works
regardless of the value's calibration — but it composed toward whatever the
value pointed at, and when that was an un-executable graze, the tree reached
it more efficiently and scored worse. Better search cannot substitute for a
better-calibrated value; it multiplies whatever value it is given.

## 5.3 Inference-time compute versus learning

The dissertation's sharpest practical claim sits at the boundary between two
strategies for improving an agent: spend more computation per decision
(search, guidance, selection — all inference-time), or change what the models
learned (more or better data, online interaction). On sub-optimal offline
data the study draws the line between them concretely. Every inference-time
lever tested — better values (the posedness ladder, correlation 0.39 → 0.98),
guidance at two noise resolutions, tree search to depth, and non-learned
selection over the generator's own beyond-ceiling imaginations — improved
reliability *up to* the best demonstrated behaviour and stopped there. The
boundary is enforced twice, independently: in the values (no learned
evaluator can prefer a plan outside its label range) and in the executable
manifold (the one non-learned evaluator finds the generator's rare
above-ceiling imaginations, but they are dead-ends or grazes that do not
survive execution). This is the inference-time counterpart of the well-known
result that return-conditioned supervised methods cannot extrapolate beyond
dataset returns — and it is arguably the more consequential finding for a
field currently investing heavily in test-time scaling: on this class of
problem, inference-time compute buys *reliability*, not *capability*. Moving
the ceiling requires moving the data. The DV authors' own mixed-versus-partial
gap (73.6 vs 94.0, identical environment and architecture, different data) is
the intervention form of exactly this claim, and the raw-data verification
(mixed demonstrations top out at three-of-four; partial's 3.1% reach four)
supplies its mechanism.

## 5.4 A note on the negative results

Three of the study's most-cited findings are negatives: naive MCTS loses to
flat selection, the tree never beats DV in absolute terms, and no lever
crosses the kitchen ceiling. It is worth being explicit that these are
*informative* negatives, not null experiments. Each is a measurement with a
mechanism and a magnitude — the −5.05 that decomposes into four named defects,
the −2.16 that is the signature of a saturated evaluator, the ceiling that is
verified on both the data side and the behaviour side. A field that rewards
positive benchmark deltas tends to under-report exactly this kind of result;
the counter-argument is that the *conditions* under which search helps are
more reusable than another point of benchmark score, because they tell a
future practitioner whether to bother.

## 5.5 Limitations

1. **Confirmed vs directional claims.** The confirmed results — DF-tree +9.04
   (maze2d, 5 seeds), +10.5 (kitchen, 4 seeds), the CG flat lift at w=8 (+4.67,
   3 seeds), the census and its data-side verification — carry multi-seed
   statistics. The guided-*tree* pin (70.0 across guidance strengths) and the
   grounded closed-loop result are single-seed (n=25) and are reported as
   directional; seed replication would firm them, though none is load-bearing
   for the law.

2. **The compute-matched control is now in place on both environments.** The
   start-matched wide-MCSS control (kitchen k600; maze2d k256) has run on both:
   on kitchen k600 spends more windows than the tree and scores below it, and on
   maze2d the DV critic-tree loses −2.16 (t=−2.60) to compute-matched k256 while
   width saturates (k50 199.4 → k256 201.2) — the tree reaches the ceiling by
   efficient sampling, not composition beyond it. This was previously listed as a
   missing maze2d control; it is closed.

3. **Two environments, not three.** Antmaze is calibration-only
   (locomotion-capped), so the law is tested on maze2d and kitchen. Whether it
   holds on, e.g., the OGBench long-horizon tasks where MCTD reports gains is
   an open and answerable question (§5.6).

4. **Evaluator caveats carry through.** The DV base critic is near-memorised on
   its training data; absolute magnitudes of critic-side MSE ratios (the
   seam ratio) are directional. All within-backbone comparisons share the
   evaluator, so orderings are unaffected, but the caveat is real.

5. **A lost provenance item.** The w=4 guided-tree run's raw per-rollout vectors
   were overwritten by a filename collision; its summary statistics survive but
   the paired vectors do not.

6. **The boundary is empirical-plus-mechanism, not a theorem.** It is possible
   in principle that a non-learned evaluator combined with a generator that
   composes past the ceiling *and* a policy that executes the composition could
   cross it; the study shows that the specific realisation of that idea
   available offline (grounded selection over DF imaginations) does not, and
   explains why (dead-end manifold, transient grazes).

## 5.6 Future work

The boundary result does more than close a door; it names the intervention
that would open it. In rough order of ambition:

- **Hindsight-stitched synthetic data (the minimal intervention).** The
  diagnostic localised the ceiling to a specific bottleneck: the 3-done
  manifold is a dead end because no demonstration continues past it. Stitching
  3-done states to 4th-task segments drawn from *other* demonstrations (the
  4th subtask is executed in isolation elsewhere in the mixed data) would
  manufacture exactly the labels the dataset lacks, staying fully offline while
  repairing the one manifold that blocks the crossing. This is the most direct
  test of whether the ceiling is truly the data's or merely this dataset's
  *arrangement*.

- **A sustained-proximity grounded value.** The grounded closed-loop failure
  was caused by union-over-steps scoring rewarding transient grazes. A value
  that requires an element to remain within threshold for k consecutive steps
  (matching the environment's own completion semantics more closely) would
  suppress the un-executable grazes; whether it then helps or simply reverts to
  the critic's behaviour is an informative experiment either way.

- **Grounded value as a leaf evaluator inside the tree, not a selector.** The
  present integration replaces the node value wholesale; using the grounded
  count only to *break ties* among critic-equal plans (rather than as the
  primary signal) would keep the critic's executability calibration while
  admitting the 4th-subtask preference where it exists — a more conservative
  combination than the blend tested.

- **The law on a third environment.** Running the DF-tree and guidance stack on
  the long-horizon OGBench tasks MCTD uses would test clause (2) at the
  high-headroom end directly, and would let the two bodies of work — this
  project's negative regime and MCTD's positive one — be measured on common
  ground rather than inferred to be complementary.

- **Online fine-tuning from the offline optimum.** Any environment feedback
  breaks the label cap; the interesting question is how *few* online
  interactions, seeded from the offline stack sitting exactly at the
  demonstration ceiling, suffice to cross it — a measurement of the value of
  the ceiling as a starting point.

## 5.7 Conclusion

Structured search over a diffusion planner is neither the free capability
amplifier that inference-time-scaling enthusiasm suggests nor the failure that
a single benchmark comparison against a saturated baseline would imply. It is
a conditional tool with a measurable operating range: it pays exactly where
its expansion is a faithful conditional generation and the flat baseline
leaves evaluator headroom, it equalises across backbone quality rather than
amplifying it, and — like every inference-time method — it is bounded by what
the data taught, a ceiling this project verified from both the data side and
the behaviour side and traced to its mechanism. The two contributions most
likely to outlast the specific systems studied are the per-token noise-aware
guidance formulation (pushing the guidance value model to the resolution a
causal sampler requires) and the boundary itself: on sub-optimal offline data,
inference-time compute buys reliability up to the best demonstrated behaviour,
and passing it is a data problem, not a search problem. The engineering
question the project started with — can search beat flat inference — turned
out to be the wrong question; the right one, which the project answers, is
when it can, by how much, and what stops it.
