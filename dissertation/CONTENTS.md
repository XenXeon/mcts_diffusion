# Planned contents — MSc dissertation (EEEM004)

*To be agreed with the supervisor before full write-up, per the module guidance.
Page estimates are indicative; total ≈ 68pp excluding appendices.*

**Working title:** *When Does Search Help a Diffusion Planner? Expansion Fidelity,
Headroom, and the Demonstration Ceiling in Offline Reinforcement Learning*

---

## Abstract (0.5pp)

## 1. Introduction (4pp)
- 1.1 Background and context — planning as generative modelling; the tension between
  inference-time scaling and the strongest published planner using no search
- 1.2 Scope and objectives — the six objectives, and the tasks undertaken to meet them
- 1.3 Achievements
- 1.4 Overview of the dissertation

## 2. Background Theory and Literature Review (14pp)
*Theory first, so a non-specialist reader can follow every later chapter; then the state
of the art, positioned to expose the gap.*
- 2.1 Offline reinforcement learning — MDP formulation, the offline setting,
  distributional shift, and why evaluation is bounded by the behaviour data
- 2.2 Denoising diffusion models — forward and reverse processes, the ε-prediction
  objective, DDPM and DDIM sampling, the ᾱ noise schedule
- 2.3 Diffusion models as trajectory planners — planning as conditional generation;
  the planner / critic / inverse-dynamics decomposition; model-predictive control
- 2.4 Steering generation — classifier guidance derived as an ε-shift, classifier-free
  guidance, and the notion of a noise-aware value model
- 2.5 Monte-Carlo tree search — selection, expansion, evaluation, backup; UCT and the
  UCB1 bound; search as policy improvement over a learned model
- 2.6 Selection under uncertainty — the optimizer's curse and maximisation bias
- 2.7 Benchmarks and metrics — D4RL, normalised score, the maze2d camping metric, the
  FrankaKitchen subtask score, and what each can and cannot measure
- 2.8 State of the art — diffusion planners; guidance; per-token-noise sequence models
  and few-step sampling; search over diffusion plans (MCTD); data ceilings in offline RL
- 2.9 Summary and the gap addressed by this work

## 3. System, Methods and Experimental Protocol (14pp)
*Specification, not chronology: what was built and how it is evaluated.*
- 3.1 Problem statement and notation
- 3.2 The base system — the frozen planner, critic and inverse-dynamics policy; the flat
  sample-and-rank inference rule
- 3.3 The tree search — node definition, composed-window scoring, UCB descent, tempered
  backup, batched forest execution
- 3.4 Expansion mechanisms — seam-glue, replacement inpainting, and exact prefix
  conditioning
- 3.5 Node value functions — the trajectory critic, distilled plan-value, and the
  goal-conditioned pessimistic value
- 3.6 The Diffusion-Forcing planner — training objective, scheduling-matrix sampling,
  and the deliberate deviations from the original instantiation
- 3.7 The shortcut-forcing planner
- 3.8 The per-token noise-aware value model and its use as guidance
- 3.9 The grounded subtask evaluator
- 3.10 Experimental protocol — pairing, start-matching, seed policy, statistical
  standards, quality gates, and the diagnostic firewall
- 3.11 Implementation and reproducibility

## 4. When Search Fails: the Full-Sequence Backbone (10pp)
- 4.1 The naive result and why it is a comparison of evaluators
- 4.2 Ill-posed node values, and the posedness falsification
- 4.3 Incomparable node scores, and composed-window scoring
- 4.4 The winner's curse: measurement, mechanism, and the price of the fix
- 4.5 Expansion infidelity: replacement inpainting and its diagnosis
- 4.6 Width, depth and the evaluator's selection ceiling
- 4.7 The limit of the value lever — the goal-conditioned pessimistic value across the
  maze2d family
- 4.8 Summary: what binds search on a full-sequence planner

## 5. When Search Pays: Faithful Conditioning and Headroom (10pp)
- 5.1 Exact prefix conditioning and the flip in sign
- 5.2 The disappearance of the winner's-curse signature
- 5.3 The cost of causality, and what search refunds
- 5.4 The headroom curve across three backbones
- 5.5 Replication on a second environment and task family
- 5.6 Composition versus sampling volume — the compute-matched controls
- 5.7 Summary: the conditions under which search pays

## 6. Guidance and the Demonstration Ceiling (10pp)
- 6.1 Why the evaluator, not the search, caps the absolute score
- 6.2 Per-token noise-aware classifier guidance — formulation and training distribution
- 6.3 Open-loop validation and the guidance-strength sweep
- 6.4 Closed-loop effect, and guidance as a partial substitute for search
- 6.5 The demonstration ceiling — the behaviour-side census
- 6.6 The demonstration ceiling — the data-side verification
- 6.7 Probing the generator: where the ceiling binds
- 6.8 The non-learned evaluator, and why the crossing still fails
- 6.9 Summary: what inference-time compute can and cannot buy

## 7. Conclusion (6pp)
- 7.1 Summary of the work and what was achieved
- 7.2 Evaluation against the objectives
- 7.3 Limitations
- 7.4 Future work

## References (IEEE numeric, ~35–45 entries) (3pp)

## Appendices
- A. Hyperparameters and training configurations
- B. Derivations — the stitched-value segment identity; the guidance ε-shift
- C. Novelty search — databases, terms, dates, and results
- D. Full experimental protocol table
- E. Selected code listings
