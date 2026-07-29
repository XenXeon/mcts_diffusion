# Introduction (dissertation chapter draft)

Draft 2026-07-11. Companion to `methodology_report.md` (methodology) and
`results_chapter.md` (results). Citations are given in author-year form; full
identifiers are consolidated in the methodology chapter's reference list and
the (now-verified) related-work chapter.

---

## 1.1 Motivation: planning as generative modelling, and the promise of search

Offline reinforcement learning asks an agent to act well using only a fixed
dataset of previously collected behaviour — no environment interaction during
training. A productive recent answer treats *planning as conditional
generative modelling*: train a diffusion model on trajectory segments from
the dataset, and at decision time sample candidate futures from it, scoring
or steering them toward high return. Diffuser [Janner et al., 2022]
established the template; Decision Diffuser [Ajay et al., 2023] refined the
conditioning; and Diffusion Veteran (DV) [Lu et al., 2025] — the base system
of this dissertation — pushed the recipe to state-of-the-art results on the
D4RL benchmark suite with a deliberately simple inference rule: sample K
candidate trajectories *unconditionally*, rank them with a learned
whole-trajectory critic, execute the best one's first step through an
inverse-dynamics policy. DV's authors call this Monte Carlo sampling with
selection (MCSS), and their central empirical finding is disarming: this
flat, guidance-free, search-free procedure *beats* the guided alternatives
(classifier and classifier-free guidance) almost everywhere they tested.

Meanwhile, a broader current in machine learning argues that *inference-time
compute* — spending more computation per decision, usually via structured
search — is a reliable route to capability: from AlphaGo and MuZero's tree
search amplifying learned value functions [Silver et al., 2017/2018;
Schrittwieser et al., 2020] to test-time scaling in large language models.
Applied to diffusion planning, the intuition runs: a flat sampler draws K
independent guesses; a tree could instead *compose* — commit to a promising
first segment, branch alternative continuations from it, evaluate deep
plans piece by piece, and thereby reach trajectories no single flat sample
contains. Concurrent work — Monte Carlo Tree Diffusion (MCTD) [Yoon et al.,
2025] — reports exactly such gains on long-horizon point-maze benchmarks.

These two observations sit in tension. If search-over-generation is a general
capability amplifier, why does the strongest diffusion planner on D4RL use no
search at all — and why did its authors find even *guidance* mostly
unhelpful? This dissertation is an attempt to resolve that tension by
measurement.

## 1.2 The research question, and how it evolved

The project began with the direct engineering question:

> **RQ0.** Can Monte-Carlo tree search, using the planner as its expansion
> operator and the critic as its evaluator, beat DV's flat MCSS inference on
> D4RL?

The direct answer arrived early and was negative: a UCB tree with MAX backup
*lost* to MCSS on maze2d (−5.05, negative on all ten seeds), and every
intuitive repair — better
per-state values, wider roots, more rollouts — bought parity at several times
the compute, never a win. But the *pattern* of failures was informative, and
the question matured into the one this dissertation actually answers:

> **RQ.** Under what conditions does structured search help a diffusion
> planner — and when it cannot help, what exactly binds it?

The answer, developed through a ladder of controlled, single-variable
experiments across two environment families (maze2d navigation and
FrankaKitchen manipulation), three planner backbones, and two guidance
mechanisms, is a three-clause empirical law:

1. **Search helps only when tree expansion is a faithful conditional
   generation.** A full-sequence diffusion planner cannot natively condition
   a continuation on a search prefix; every workaround (gluing, replacement
   inpainting) injects artifacts that the evaluator mis-scores and the search
   then actively exploits (a measured winner's curse). Training a causal,
   per-token-noise planner (Diffusion Forcing [Chen et al., 2024]) makes
   prefix-conditioning exact — and tree search flips from harmful to helpful,
   confirmed across seeds on both environment families.
2. **The gain is proportional to the headroom the flat baseline leaves.**
   Across backbones of decreasing flat quality (DV → DF → shortcut-forcing)
   the tree's gain grows monotonically (−2.2 → +9.0 → +37.2, DV start-matched) while its
   landing point barely moves; the same curve reappears *within* one
   environment as generation-side guidance strengthens the flat pool (+8 →
   +6 → +4). Search is a partial equaliser, not an amplifier: it refunds
   what a weaker sampler loses, and pays nothing above a strong one.
3. **No learned-value method passes the dataset's demonstration ceiling.**
   On kitchen-mixed — whose 613 demonstrations never solve all four subtasks
   (verified in the raw data) — every method in an 850+-rollout census,
   spanning four families (flat selection, tree search, token-level and
   trajectory-level guidance, on both backbones), stops at or below the score
   of the best demonstrated outcome; the baseline itself sits exactly on it.
   Inference-time structure improves *reliability up to* the best behaviour
   the data teaches; passing it requires better data or a value signal that
   is not distilled from the data.

## 1.3 Contributions

1. **A mechanism-level negative result for naive search on a SOTA diffusion
   planner.** We decompose *why* MCTS loses to flat selection into four
   measured defects — ill-posed per-state values (fixed by plan-value
   distillation: correlation 0.39–0.76 → 0.98), incomparable node windows
   (fixed by composed-window scoring), evaluator exploitation by MAX backup
   (the optimizer's curse, measured at −4.29 and fixed by top-m backup, +4.54
   at roll-t 5.00), and unfaithful expansion (unfixable on a frozen
   full-sequence planner; replacement inpainting measured at −18.5) — and
   show the first three fixes together buy only parity, isolating expansion
   fidelity as the binding constraint.
2. **A confirmed positive: tree search helps Diffusion-Forcing planners.**
   With exact prefix conditioning the winner's-curse signature vanishes and
   the tree beats its own flat baseline on maze2d (+9.04, paired t = 3.90,
   5 seeds, n = 125) and kitchen (+10.5, paired t = 5.47, 4 seeds, n = 100;
   one extra subtask in 43% of rollouts) — with a compute-matched width
   control showing the kitchen gain is composition, not sampling volume.
3. **The headroom law**, measured twice independently: across three backbones
   on maze2d, and across three guidance strengths within kitchen — including
   the finding that guidance and search are partial substitutes converging
   on the same landing point.
4. **Per-token noise-aware classifier guidance.** We push the guidance value
   model from trajectory-level to token-level noise resolution — the
   diffusion-forcing property applied to the critic, required because a
   causal sampler has no single trajectory-level noise — trained on the
   sampler's actual query distribution (schedule rows with clean-history
   prefixes). It lifts the flat causal baseline monotonically in guidance
   weight (60 → 64 → 66) at zero measured physical cost, with an
   *independent* evaluator confirming the shift.
5. **A verified inference-time boundary.** The demonstration-ceiling claim is
   established from both sides — the raw-data distribution (no kitchen-mixed
   demonstration ever solves all four subtasks; the partial split's 3.1% is
   the control) and the rollout census — with the DV paper's own
   mixed-vs-partial contrast (73.6 vs 94.0) reinterpreted as the intervention
   form of the same law. A pre-registered test (the original trajectory-level
   CG, predicted ≤ 75, landed 74.75) survived its falsification opportunity.
6. **Engineering artifacts** enabling the above: a paired-evaluation harness
   over the frozen DV stack; a faithful minimal Diffusion-Forcing planner and
   a shortcut-forcing (few-step) variant sharing one checkpoint/dispatch
   scheme; the noise-aware critic and its trainer; and quality gates
   (open-loop fidelity checks, dataset-ceiling verification) that caught
   several results-corrupting defects before they reached conclusions.

## 1.4 Scope and honest framing

This dissertation does not claim a new state of the art. DV's flat inference
remains the best absolute score on both environments studied, and clause (2)
of the law explains why that is not a coincidence: DV's baseline leaves
search no headroom on these benchmarks. The contribution is the *conditions*
— when search helps, by how much, and what stops it — each clause carried by
its own controlled experiment rather than by aggregate benchmark deltas. The
positive results are relative (tree vs its own flat baseline, same planner,
same evaluator, paired rollouts); the boundary result is absolute and, we
argue, the more consequential of the two for practitioners deciding whether
to spend inference compute on search.

## 1.5 Document structure

Chapter 2 reviews diffusion planning, guidance, and search-over-learned-
models, and positions this work against MCTD and the return-conditioned
extrapolation literature. Chapter 3 (methodology) specifies the DV stack, the
tree algorithm and its instrumentation, the Diffusion-Forcing and
shortcut-forcing backbones, and the noise-aware critic, together with the
experimental protocol (paired arms, seed policy, statistical standards).
Chapter 4 (results) presents the evidence in the order of the law's clauses.
Chapter 5 discusses implications — inference-time scaling versus learning,
the equaliser view of search, and the data ceiling — together with
limitations and future work, including the one design that could in principle
pass the ceiling (a grounded, non-learned subtask evaluator).
