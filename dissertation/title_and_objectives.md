# Title and objectives — for supervisor sign-off

*Draft for agreement before full write-up. The objectives below are the ones the
Conclusion will be evaluated against, so they need to be settled first.*

---

## Title

**Recommended:**

> **When Does Search Help a Diffusion Planner? Expansion Fidelity, Headroom, and the
> Demonstration Ceiling in Offline Reinforcement Learning**

**Alternatives:**

> Structured Search over Diffusion Planners: the Conditions Under Which
> Inference-Time Compute Improves Offline Decision-Making

> Conditions and Limits of Monte-Carlo Tree Search over Diffusion Trajectory Planners

**Rationale for the change.** The original working title ("Integrating MCTS into
long-horizon planning for diffusion planners") names the *task undertaken*, not the
*contribution made*. The investigation established a set of conditions and a boundary
rather than an integration, and the title should say so — the module guidance
explicitly anticipates the title changing once the direction and findings are known.

---

## Objectives

Stated as points of advance — what new knowledge the work establishes — rather than as
tasks. Each is evaluated explicitly in the Conclusion.

- **O1.** Establish whether inference-time tree search can improve on flat
  sample-and-rank selection in a state-of-the-art diffusion planner, and identify the
  conditions that determine the outcome.
- **O2.** Determine the mechanisms by which tree search over diffusion-generated plans
  succeeds or fails, isolating the separate contributions of node-value posedness,
  node-score comparability, backup robustness under evaluator noise, and expansion
  fidelity.
- **O3.** Establish the role of the planner's conditioning mechanism, by constructing a
  causal per-token-noise planner that makes prefix-conditioned expansion an exact
  conditional generation, and quantifying the resulting change in the value of search.
- **O4.** Quantify the relationship between the gain delivered by search and the quality
  of the baseline against which it is measured, across planner backbones and across
  generation-side guidance strengths.
- **O5.** Extend classifier guidance to the token-level noise resolution that causal
  sequence samplers require, and evaluate it as a generation-side alternative to
  search.
- **O6.** Determine the limit that sub-optimal offline demonstration data imposes on
  inference-time methods, and establish that limit independently from the data
  distribution and from deployed behaviour.

### Tasks undertaken to meet them

*(Included per the template's suggestion, to convey the scale of work behind the
objectives. These are tasks, not objectives.)*

- Reproduction of the base system's published results on two D4RL environment families,
  and construction of a paired evaluation harness over the frozen stack.
- Implementation of a batched Monte-Carlo tree search operating over diffusion-generated
  plans, with three interchangeable expansion mechanisms and four node-value functions.
- Training of two additional planner backbones — a causal per-token-noise
  (Diffusion-Forcing) planner and a few-step shortcut-forcing variant — sharing one
  checkpoint and dispatch scheme with the base system.
- Training of a per-token noise-aware value model on the sampler's own query
  distribution, and its deployment as classifier guidance on the frozen causal planner.
- Construction of a grounded, non-learned subtask evaluator for imagined states.
- Approximately twenty controlled, seed-replicated closed-loop experiment arms, together
  with open-loop quality gates and a raw-data verification of the demonstration ceiling.

---

## Mapping — objectives to chapters

| Objective | Where established | Where evaluated |
|---|---|---|
| O1 | Ch 4, Ch 5 | 7.2 |
| O2 | Ch 4 (§4.2–4.5) | 7.2 |
| O3 | Ch 3 (§3.6), Ch 5 (§5.1–5.3) | 7.2 |
| O4 | Ch 5 (§5.4), Ch 6 (§6.4) | 7.2 |
| O5 | Ch 3 (§3.8), Ch 6 (§6.2–6.4) | 7.2 |
| O6 | Ch 6 (§6.5–6.8) | 7.2 |

Every objective is carried by at least one multi-seed confirmed result. O4's
within-environment instance (the guidance curve) currently rests on fewer seeds than
its cross-backbone instance; a re-run is in progress to firm it.

---

## Note on revision

These differ from the objectives in the interim report, which were framed around
integrating tree search and improving benchmark performance. The revision reflects what
the investigation established: the direct performance objective returned a negative
answer early, and the work matured into characterising the conditions and the boundary.
The module guidance permits revising objectives for the final dissertation, and advises
agreeing the revision with the supervisor — hence this document.
