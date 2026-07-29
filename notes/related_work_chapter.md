# Related Work (dissertation chapter draft)

Draft 2026-07-11 (citations verified 2026-07-13). Each section ends with a
one-line positioning statement ("this work:") so the chapter reads as an
argument, not a survey. All previously [VERIFY]-flagged references were
checked against arXiv/proceedings and are now given with full author lists and
identifiers; a final formatting pass into the dissertation's bibliography
style is all that remains.

---

## 2.1 Diffusion models as trajectory planners

Diffuser [Janner et al., ICML 2022] introduced planning-as-denoising: a
diffusion model over state(-action) trajectory segments, with plans produced
by iterative refinement and steered toward return via the gradient of a
separately trained value model (classifier guidance). Decision Diffuser
[Ajay et al., ICLR 2023] replaced the external classifier with
classifier-free guidance on a return-conditioned model, arguing conditioning
beats gradient steering. Both established the core stack this dissertation
inherits: a trajectory generator, a return evaluator, and an inverse-dynamics
model to turn planned states into actions.

Diffusion Veteran (DV) [Lu et al., ICLR 2025] is a large-scale empirical
study of what actually matters in this design space, and the strongest
published planner on the D4RL tasks studied here. Its conclusions shape this
project twice over. First, its architecture: a full-sequence transformer
denoiser over stride-spaced waypoints, a whole-trajectory critic, and a
diffusion inverse-dynamics policy — all frozen and reused unchanged as this
dissertation's substrate, so that every comparison isolates the *inference
mechanism* rather than the models. Second, its headline finding: flat
sample-and-rank (MCSS) beats guided generation nearly everywhere, with
FrankaKitchen — where most demonstrations are sub-optimal — as the one
partial exception, motivating their hypothesis that guidance pays exactly
when the dataset lacks near-optimal behaviour. That hypothesis is a direct
ancestor of this dissertation's kitchen arc and its demonstration-ceiling
result, which sharpens it: what binds is not "sub-optimality" in the
abstract but the *best demonstrated outcome*, which no learned-value method
was observed to pass — and which their own mixed-vs-partial contrast (73.6
vs 94.0) tracks exactly.

**This work:** keeps DV frozen as the measured substrate, and asks what
inference-time structure can add on top of it.

## 2.2 Guidance: classifier, classifier-free, and the resolution of noise

Classifier guidance (CG) steers diffusion sampling with the input-gradient of
a model trained to predict a label from *noised* inputs [Dhariwal & Nichol,
2021]; classifier-free guidance (CFG) bakes the condition into the denoiser
and extrapolates between conditional and unconditional predictions
[Ho & Salimans, NeurIPS 2021 Workshop on Deep Generative Models; arXiv:2207.12598].
In diffusion RL, Diffuser's value
guidance is CG with return as the label; Decision Diffuser is the CFG
counterpart. In all of these, the guidance value model is noise-aware at
**trajectory level**: one noise scalar per window, matching full-sequence
samplers whose every token shares a noise level.

Diffusion Forcing breaks that symmetry: under per-token noise there is no
single "noise level of the trajectory" — at any sampling step the history is
clean while the far future is near-noise. This dissertation's noise-aware
critic V(x, k) conditions on the **per-token noise vector** — the
diffusion-forcing property applied to the value function — and is trained on
the sampler's actual query distribution (schedule rows with clean-history
prefixes mixed with uniform coverage), then applied as an eps-shift CG on the
frozen causal planner. To our knowledge the token-level-resolution guidance
value model is new; the closest-sounding prior work, Diffusion Actor-Critic
[Fang et al., 2024, arXiv:2405.20555], applies Q-gradients inside policy
*training* and trains no noise-conditioned value model. The original
trajectory-level CG is retained in this dissertation as a comparison arm and
as a pre-registered test of the ceiling claim (it landed at parity with MCSS,
as predicted).

**This work:** pushes the guidance value model from trajectory-level to
token-level noise resolution, and measures both resolutions against the same
frozen stack.

## 2.3 Sequence diffusion with per-token noise, and few-step sampling

Diffusion Forcing (DF) [Chen et al., NeurIPS 2024] trains a causal denoiser
where every token carries an independent noise level, unifying
teacher-forcing-style training with diffusion sampling. Two of its properties
are load-bearing here: (i) clean-history + noisy-future inputs are
*in-distribution by construction*, making conditioning a continuation on a
search prefix an exact conditional generation — precisely the capability
whose absence this dissertation measures as the binding failure of tree
search on full-sequence planners; (ii) its scheduling-matrix sampler admits
"causal uncertainty" schedules (near future resolves before far future). Our
implementation (methodology §7.2) deliberately deviates from the paper's
instantiation — transformer DiT blocks with per-token adaLN, critic-ranked
selection instead of their guidance schemes — so that the DF arm differs from
the DV arm *only* in the planner.

Shortcut Models [Frans et al., 2024, arXiv:2410.12557] train the denoiser to
be consistent across dyadic step sizes, enabling sampling in 1–2^k large
steps; the Dreamer 4 world model [Hafner, Yan & Lillicrap, 2025, "Training Agents
Inside of Scalable World Models", arXiv:2509.24527] combines this with per-token
noise ("shortcut forcing") for real-time imagination. This
dissertation's shortcut-forcing planner follows that recipe and serves a
specific scientific purpose: a *deliberately weaker/faster* third backbone
that supplies the third point on the headroom curve — and a caution, since
its plans score above the 52-step planner's under the learned critic while
executing worse (a value-model/controllability dissociation relevant to any
few-step sampler used inside model-predictive control).

**This work:** uses DF for exact prefix conditioning, and shortcut-forcing as
a controlled weak-backbone probe — both slotted into the identical harness.

## 2.4 Search over learned models, and its failure modes

The modern search-amplifies-learning template comes from AlphaGo/AlphaZero
[Silver et al., 2017/2018] and MuZero [Schrittwieser et al., 2020]: UCT-style
selection [Kocsis & Szepesvári, 2006] over a learned model, with a learned
value at the leaves, improving both play and (via distillation) the networks
themselves. This dissertation's Lever B — distilling the trajectory critic's
plan values into a per-state value — is the offline reduction of that
distillation step, and its success (correlation 0.98, +35 closed-loop)
against the failure of directly-regressed V(s) is explained by target
*posedness* rather than capacity, an observation we believe is
underappreciated in the offline-RL value literature.

Search also inherits a classical pathology: selecting on noisy estimates
biases the selected value upward — the optimizer's curse [Smith & Winkler,
2006], known in MCTS as maximisation bias. This dissertation contributes a
planner-specific mechanism and measurement: on stitched (glued) expansions
the evaluator's error is not merely noisy but *systematically exploitable*
(the seam is off-manifold; MAX backup promotes exactly the plans that fool
the critic, measured at −4.29), the tempered top-m backup prices the fix
(+4.54, roll-t 5.00), and — diagnostic of the mechanism — the curse signature
*vanishes* once expansion is exact (MAX ≈ top-3 under DF).

**This work:** locates the curse's fuel in expansion infidelity, not in
search per se, and shows the backup fix becomes unnecessary exactly when the
expansion is faithful.

## 2.5 Tree search over diffusion plans: MCTD and the bracketing of "when"

Monte Carlo Tree Diffusion (MCTD) [Yoon et al., 2025, arXiv:2502.07202]
frames denoising itself as tree-structured rollout — branching over
"guidance levels" on trajectory chunks, with jumpy denoising as a fast
rollout — and reports strong gains on long-horizon OGBench/point-maze tasks.
MCTD is the concurrent *positive* pole of the question this dissertation
brackets, and the two are complementary rather than competing: MCTD
demonstrates that search-over-diffusion *can* pay handsomely; this
dissertation measures *when* — its benchmarks (long horizons far exceeding
the planning window, mazes demanding composition) sit squarely in the
high-headroom regime our law identifies, while its baselines are far from a
DV-grade selection ceiling. The law predicts both results at once: MCTD's
wins where flat baselines are weak and horizons exceed windows, and our
nulls where a saturated critic already extracts the dataset's best mode.
Notably, MCTD uses no learned value ranking of the kind DV's critic provides;
our kitchen result suggests its gains, too, would compress against a
selection-saturated baseline.

**This work:** supplies the negative regime, the conditions, and the ceiling
that together turn "search helps diffusion planners" from an existence claim
into a law with scope conditions.

## 2.6 Data ceilings in offline RL

That offline methods are bounded by their data is folklore; the sharp version
matters. Return-conditioned supervised methods (Decision Transformer family)
are known to fail to extrapolate beyond dataset returns [Brandfonbrener,
Bietti, Buckman, Laroche & Bruna, "When does return-conditioned supervised
learning work for offline reinforcement learning?", NeurIPS 2022,
arXiv:2206.01079; Emmons, Eysenbach, Kostrikov & Levine, "RvS: What is
Essential for Offline RL via Supervised Learning?", ICLR 2022,
arXiv:2112.10751] — the conditioning analogue of this dissertation's claim.
The D4RL kitchen splits [Fu et al., 2020; env from Gupta, Kumar, Lynch, Levine
& Hausman, "Relay Policy Learning", CoRL 2019, arXiv:1910.11956] encode the
distinction by construction: *partial* contains complete-task demonstrations,
*mixed* does not. This dissertation adds the inference-time counterpart with unusual
directness: a raw-data verification (no mixed demonstration ever has all
four goal subtasks solved; the best mode, 3-of-4, appears in 43% of
demonstrations) paired with an 850+-rollout census in which four method
families — flat selection, tree search, and guidance at both noise
resolutions — all stop at or below that best demonstrated outcome, the
strongest baseline sitting exactly on it (100% extraction of the best mode).
The mechanism is stated at the level of the value function: every learned
evaluator's targets top out below the undemonstrated outcome, so no amount
of inference-time optimisation *against those evaluators* can prefer it.

**This work:** verifies the ceiling on both the data side and the behaviour
side, identifies the one lever exempt from it (a grounded, non-learned
evaluator — future work), and reframes DV's mixed-vs-partial gap as the
intervention form of the same law.

## 2.7 Summary of positioning

| axis | prior work | this dissertation |
|---|---|---|
| base system | DV: flat MCSS is SOTA | kept frozen; inference mechanism is the only variable |
| search over diffusion | MCTD: positive, high-headroom benchmarks | negative regime + conditions + ceiling; paired, seeded, mechanism-level |
| guidance | CG/CFG at trajectory-level noise | token-level noise-aware CG; both resolutions measured on one stack |
| backbone | full-sequence planners | DF for exact prefix conditioning; shortcut-forcing as weak-backbone probe |
| failure analysis | optimizer's curse (generic) | curse fueled by expansion infidelity; vanishes under exact conditioning |
| data limits | return-conditioning can't extrapolate | inference-time counterpart, verified data-side + census-side |
