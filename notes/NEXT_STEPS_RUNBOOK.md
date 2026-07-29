# RUNBOOK — from here to the dissertation and a paper

*Written 2026-07-06, HANDOFF REFRESHED 2026-07-11. Self-contained: assume the reader
(human or agent) has NO conversation context. Read `notes/methodology_report.md` for
the science; this file is the TODO list with exact commands, expected outputs, and
decision rules. All commands run from the repo root on the GPU box (/workspace);
the local Windows box is torch-free (tests/py_compile only). Scores: DV camping
metric on maze2d; 0-100 subtask score on kitchen (25 pts = 1 subtask).*

## ★ STATE OF THE PROJECT — HANDOFF SNAPSHOT (2026-07-11) ★

**Phase: WRITING. The experimental program is complete except two optional items.**

CONFIRMED results (do not re-run; full numbers in §1 + notes/results_chapter.md):
- maze2d: DV-tree never beats DV-MCSS (parity at 6x compute; critic-ceiling ~206).
  DF-tree beats DF-MCSS +9.04 paired t=3.90 (5 seeds, n=125). Shortcut backbone:
  +37.2 t=6.58 -> 3-point headroom curve (tree = equalizer).
- kitchen-mixed: DV-MCSS reproduces paper at 75.0 (= EXACTLY 3-of-4 every rollout).
  DV-tree NULL. DF-tree +10.5 paired t=5.47 (4 seeds, n=100). Guided (per-token
  noise-aware CG, the novel artifact): flat lift CONFIRMED at w=8 (+4.67, paired
  t=2.22, n=75, 3 seeds); w=4 NOT separable (+1.50, t=0.46) — claim w=8 only.
  Guided tree pinned at 70.0 across w in {0,4,8} (single-seed each).
- THE BOUNDARY (both sides verified): kitchen-mixed demos NEVER reach 4-of-4
  (max-solved histogram {0:1,1:35,2:312,3:265}); 850+ rollout census across 4
  method families has ZERO 4-task completions; DV-CG (original trajectory-level
  CG, pre-registered <=75) landed 74.75. 75.0 = the demonstration ceiling.
- Dissertation drafts DONE in notes/: methodology_report.md, results_chapter.md,
  intro_chapter.md, related_work_chapter.md, discussion_chapter.md. All 6 FIGURES
  generated: scripts/make_figures.py -> figures/fig1..6_*.{pdf,png} (rerun to
  refresh from results/). T1 protocol table DONE (results_chapter §11, compute
  from JSON wall_s). All 6 [VERIFY] citations RESOLVED 2026-07-13 (Ho&Salimans
  NeurIPS2021 WS/arXiv:2207.12598; Fang et al. 2405.20555; Dreamer4 =
  Hafner/Yan/Lillicrap arXiv:2509.24527; Brandfonbrener NeurIPS2022/2206.01079;
  Emmons RvS ICLR2022/2112.10751; Gupta RelayPolicyLearning CoRL2019/1910.11956).
  Remaining docs: bibliography-style formatting pass + moving the .md chapters
  into the submission format (LaTeX/Word). The experimental program is COMPLETE.

IN FLIGHT / NEXT ACTIONS (ordered):
1. GROUNDED CLOSED LOOP — DONE 2026-07-12, crossing FAILED (informatively).
   kitchen_both_df_grounded_s0.json: grounded flat 55.0+/-2.8, tree 63.0+/-3.2
   (tree +8.0 over its own flat preserved), NO rollout hit 100. Grounded value
   scores BELOW its critic-valued counterparts (flat 60, tree 68-70) — the
   2nd imagination/execution dissociation (after shortcut): grounded ranks by
   imagined touched-count, rewarding the transient near-goal GRAZES the union
   scoring admits; preferring a graze over the critic's executable pick
   diverts the executed step toward completions that never cash, and the
   coarse 5-level score dithers across replans. The critic's conservatism is
   load-bearing for execution. => BOUNDARY ENFORCED TWICE (label cap in the
   values + executable-manifold cap on the generator's rare 4-imaginations).
   The moonshot is CLOSED as an offline lever; write it as the boundary's
   final exhibit. What would change the answer changes the problem (partial
   data / online / hindsight-stitch 3->4 synthetic data) = future work.
   DO NOT iterate grounded selection tricks (sustained-proximity scoring,
   blend tuning) unless the WRITE-UP is done and there is spare time — it is
   future-work material, not a result to chase now.
2. Optional: DV-CG w-scan completion (w0.5/1.0 re-runs with stdout redirection).
3. WRITE: discussion/conclusion chapter + figures. THE ONLY REMAINING WORK —
   the experimental program is complete; every lever has been tested vs the
   ceiling and the ceiling held.
DO-NOT list: no partial-v0 stack training (cite DV's 73.6-vs-94.0 instead); no
more w-sweeps/deeper trees/backbones chasing >75 (label-cap: provably capped);
no re-running any seed set marked confirmed. Everything is still UNCOMMITTED in
git (user commits manually) — remind them.

*** V(s,g) VALUE-LADDER RESULT (2026-07-13, RESOLVED — refines the law, NOT a
tree win) ***: enabled V(s,g) on maze2d, ran the goal-conditioned trees.
10-seed means: critic(top-3) ~202, V(s) 202.5, V(s,g) 205.1, V(s,g)-pess 205.2.
The ladder CLIMBS with value posedness (goal-conditioning +3, pessimism the
best) toward but not past the MCSS ceiling. *** UPDATED 2026-07-13 with START-MATCHED k256 (the decisive baseline) ***:
the old k256=206.0 was ALSO a cross-start artifact (same trap as 204.9). The
START-MATCHED k256 (maze2d_large_mcss_k256_s0-9) = 201.2, and k50->k256 gains
only +1.8 on matched starts (width saturates). So V(s,g)-pess (205.2, 10 seeds)
BEATS the compute-matched wide MCSS: **+3.98, seed-t=2.83, per-rollout t=2.56,
start-matched, 10 seeds** — a GENUINE DV-backbone tree WIN (width saturation
rules out "just more samples": tree ~290 samples, k256=256, k256 still loses
by 4). Ladder: V(s,g)-pess WINS (+4 vs k256), V(s,g)-mean PARITY (+1.8 t=0.99,
5 seeds), critic/V(s) parity/loss => PESSIMISM (ensemble-min) is the
load-bearing ingredient (winner's-curse insurance on selection).
CONCLUSION (REVISED — this overturns "value lever exhausted / DV-tree never
wins"): the value lever WORKS on the frozen DV backbone, but ONLY with a
well-posed pessimistic goal-conditioned value; ill-posed values (critic, V(s))
and the goal-conditioned MEAN don't cross the ceiling. This is the project's
ORIGINAL goal achieved (MCTS beats MCSS on the SOTA planner) — needs careful
integration into the dissertation, NOT a footnote. Also answers the owed maze2d
k290 compute-matched control (start-matched k256 IS it). DF-tree win untouched.
GOTCHA (recorded, cost 2 flip-flops): on maze2d the camping metric is so
start-sensitive that ONLY start-matched comparisons are valid — never compare
tree at seeds 0-9 to a baseline measured on other starts (204.9, old k256=206
were both cross-start). analyze_maze2d_values.py pairs on the 'starts' arrays.
*** GENERALIZATION FAILS — the maze2d-large win is an OUTLIER, NOT robust
(2026-07-13). DO NOT headline it / DO NOT rewrite the dissertation around it. ***
V(s,g)-pess vs compute-matched k256, start-matched, per env:
  maze2d-large  205.2 vs 201.2 = +3.98  seed-t=+2.83  (10 seeds)  WIN
  maze2d-medium 144.2 vs 159.0 = -14.82 seed-t=-2.69  (5 seeds)   LOSS
  maze2d-umaze  135.3 vs 140.2 = -4.90  seed-t=-3.06  (5 seeds)   LOSS
So the SAME method LOSES on 2 of 3 maze2d sizes. The project's ORIGINAL
narrative ("value lever exhausted / DV tree doesn't beat MCSS") is likely
CORRECT and now BETTER supported (3 sizes: 2 loss + 1 outlier). The large +4
is probably a start-set/geometry artifact.
DIAGNOSTIC RESOLVED 2026-07-13: all three V(s,g) values are WELL-TRAINED —
val_corr large 0.875, medium 0.801, umaze 0.621. Medium's value is strong
(0.80) yet its tree LOSES by -14.8 => the losses are REAL, not undertrained-
value artifacts. The maze2d-large win is a genuine OUTLIER (1 of 3 sizes),
likely a start-set/geometry artifact of large's long corridors. CONCLUSION:
value posedness does NOT yield a reliable tree gain on the DV backbone — this
CONFIRMS "the value lever is exhausted", it does not overturn it. The drafts'
"no win on DV" narrative is CORRECT; NO RQ reversal, NO headline win.
DONE 2026-07-13 (fixes 1+2+3): (1) start-matched measurement note added to
methodology §7.5 † (metric is start-sensitive; DV-MCSS k50 199.4 / k256 201.2;
DV critic-tree loses −2.16 t=−2.60 to compute-matched k256, reproducing −2.6;
204.9 = compute-matched-scale on seeds-0-2); ‡ notes on the headroom tables
(methodology + results); intro/discussion −2.6→−2.2 (start-matched); F3
regenerated (DV 201.2/199.0). (2) Limitation #2 (k≈290 control) RETIRED in
methodology §9.2 / results §9.2 / discussion §5.5 — start-matched k256 IS the
control, DV tree loses to it. (3) results_vsg_ladder_draft.md REWRITTEN as
CONFIRMING evidence (value posedness gives no reliable DV-tree gain: large win
+3.98 is an outlier, medium −14.8 / umaze −4.9 losses, all well-trained values
val_corr 0.62-0.88) — supports "value lever exhausted", NOT a headline win.
NOTE: bare 204.9 remains in phase-1 narrative prose (methodology 74/156/174,
results 42/48, T1) — covered by the convention note; left to avoid over-churn.
DO NOT (pending the val_corr diagnostic): rewrite intro/results/discussion to
make V(s,g)-pess the RQ0 answer; draft the value-ladder headline subsection;
run the mean/k50 arms on medium/umaze. results_vsg_ladder_draft.md is on HOLD.
204.9 PROVENANCE (resolved): 204.9 is the methodology's reference DV-MCSS k50
(n=150) measured on a DIFFERENT start set than seeds 0-9; the maze2d camping
metric is strongly start-dependent (notes/baseline_seed_variance.md), so
204.9 != the on-disk seed-0-9 k50 (199.4) — different starts, not an error.
=> the "-2.6 critic-tree gap vs 204.9" is a CROSS-START artifact; recompute all
maze2d tree-vs-MCSS deltas START-MATCHED (analyze_maze2d_values.py pairs on the
'starts' arrays). The DF/shortcut headroom rows are paired within-file so they
are unaffected; only the DV-row magnitude and F2/F3's DV baseline need the
start-matched number (qualitative "no DV critic-tree win" holds either way).
ANALYZER CAVEAT: analyze_maze2d_values.py pools ALL critic-tree files (mixed
MAX/top-3/stitched, 10 seeds -> −3.88) — its critic/V(s) rows MIX configs; trust
only the clean single-config v_sg / v_sgpess rows there.
TO FINISH CLEANLY (optional, for a symmetric table): run a START-MATCHED wide
MCSS (k256 or k290) at seeds 0-9 so the compute-matched comparison is paired,
not cross-start. Then the V(s,g) value-ladder can go in the methodology as the
"value lever exhausted at its well-posed limit" result + the k290 control.

V(s,g) SCOPE (2026-07-13): V(s,g) is a SPATIAL time-to-xy-goal value (goal =
obs[:,:2], target = normalised time-to-reach; mcts/relabel.py). It applies to
the NAVIGATION envs only. antmaze: trained (V(s) 10 seeds, V(s,g)_pess 3).
maze2d: ENABLED 2026-07-13 — the maze2d seq dataset now exposes seq_tml (its
learn_policy=False paths end at a goal-reach = a genuine terminus; verified
round-trip through relabel.path_end_indices). Train + run:
  python scripts/train_state_value.py --env maze2d-large-v1 --goal-conditioned
  (saves state_value_sg_ckpt_best.pt; all maze2d paths are terminus-reaching so
   default terminus-only mode = full-data mode here)
  for i in 0 1 2 3 4; do python scripts/run_mcts_compare.py --env maze2d-large-v1 --method mcts --value-mode v_sg --sg-ckpt state_value_sg_ckpt_best.pt --k-root 50 --top-m 3 --budget 15 --k-mcts 16 --n-envs 25 --n-episodes 1 --seed $i --out results/m2l_tree_vsg_s${i}.json; done
  (--method MCTS not both: run_mcts_compare guards method in {mcss,both} to
   value_mode in {v_s,critic,grounded} — v_s/v_sg/v_sg_pess trees are mcts-only,
   compared vs the shared DV-MCSS baseline. v_sg_pess = ensemble-min variant.)
kitchen: V(s,g) does NOT apply (obs[:,:2] = joint angles, not a spatial goal;
the goal is a subtask set) — the guard in train_state_value now says so. Report
V(s,g) for the nav envs and state this scope in the methodology (nav vs
manipulation), NOT as a missing cell. Prior oracle evidence (geodesic-greedy 80
< DV-MCSS 107 on maze2d) predicts V(s,g) tree <= MCSS: completeness, not a win.

REPO LAYOUT (tidied 2026-07-13): scripts/ holds the 13 LIVE scripts (see
scripts/README.md) + scripts/legacy/ (38 archived Phase 0-6 / closed-lever
one-offs, scripts/legacy/README.md). mcts/ kept FLAT (module map in
mcts/__init__.py). No files renamed among the live set — all runbook commands
below are unchanged. 124 torch-free tests green post-tidy.

---

## 0. Shell discipline (a real run was lost to this — read first)

- **Never paste multi-line backslash-continued commands into a bash loop.** Windows
  CRLF mangles `\`-continuations; bash then executes only the first line and treats
  the rest as separate commands (`bash: --df-ckpt: command not found`). Write loop
  bodies as ONE line.
- **Every valid DF run prints `loaded DF planner: df_planner_ckpt_<tag>.pt` at
  startup** (from load_models). If that line is missing from a run that should use
  `--df-ckpt`, the run is a DV-backbone run — discard it.
- **Every valid run ends with `saved results -> <path>`.** No line = nothing saved.
- Runs are deterministic given `--seed`: two runs with the same seed+config produce
  bit-identical numbers. Identical outputs across "different seeds" means the seed
  flag never reached the script.
- Known-invalid artifacts: the 2026-07-06 "seed 1/2" loop outputs (201.7 / 197.9,
  n=50) were default-seed DV-backbone duplicates with no `--out`. Ignore them.

## 1. Verified state (do not re-run these)

maze2d-large-v1, DV camping score. Tree arms: budget 15, k_mcts 16, child_index 1.

**START-MATCHED as of 2026-07-28** — all DV deltas below are differenced against the
MCSS arm on the SAME seeds, `starts` asserted equal. Full audit + old→new mapping:
`notes/maze2d_startmatched_correction.md`. **Never difference arms from different
files without checking `starts` first.**

| arm | score | Δ vs matched MCSS | n / seeds | file(s) |
|---|---|---|---|---|
| DV MCSS k50 | 199.4 (s0-9) / 202.0 (s0-2) | — | n=250 | results/maze2d_large_mcss_k50_s*.json |
| DV MCSS k256 | 201.2 (s0-9) / 204.4 (s0-2) | +1.8 vs k50 (5× samples — saturated) | n=250 | results/maze2d_large_mcss_k256_s*.json |
| DV naive tree (orig cfg) | 194.3 | **−5.05, seed-t −10.57** (neg. all 10 seeds) | n=250 s0-9 | results/maze2d_large_critic_tree*.json |
| DV tree r50 MAX (glue) | 197.7 | **−4.29, seed-t −4.75** | n=75 s0-2 | results/m2l_tree_r50_s*.json |
| DV tree r50 top-3 (glue) | 202.2 | **+0.25 t=0.33 vs k50 (parity)**; −2.16 t=−2.60 vs k256 | n=75 s0-2 | results/m2l_tree_r50_m3_s*.json |
| MAX → top-3 (curse fix) | — | **+4.54, roll-t 5.00** (same starts, baseline cancels) | n=75 s0-2 | both of the above |
| DV tree inpaint r50 MAX | 182.1±11.1 | **−18.53, t=−3.02** | n=25 s0 | results/m2l_tree_criticr50_inpaint.json |
| planv-tree (V(s)=plan-value) | 202.5 | +0.52, t=0.30 (parity) | n=75 s0-2 | results/m2l_tree_planv_s*.json |
| stitched-critic MCSS (Lever A) | 200.8 | +0.16, t=0.12 (**null, not a harm**) | n=25 s0 | results/m2l_mcss_k50_stitched_s0.json |
| **DF MCSS k50** | **182.7±9.5** | n=25 s0 | results/m2l_mcss_df_s0.json + both-file |
| **DF tree r50 top-3** | **192.2±8.2 (+9.4, paired t=1.53)** | n=25 s0 | results/m2l_both_df_m3_s0.json |
| **DF tree r50 MAX** | **190.4±8.1 (+7.6, t=1.35)** | n=25 s0 | results/m2l_tree_df_max_s0.json |

DF checkpoints: maze2d-large `df_planner_ckpt_final.pt` (good: gen hops == real,
critic gap 0.065); antmaze-large-diverse `df_planner_ckpt_final.pt` (weak: DF-MCSS
reach 44% vs DV 76.9%, teleport tail — calibration only, do NOT retrain).

Headline hypothesis — **CONFIRMED 2026-07-07 (seeds 0–4, n=125, pooled paired t=3.90,
+9.04):** on the DF backbone the tree BEATS its own flat baseline and the winner's-curse
signature (MAX≪top-m) vanishes. The shortcut-forcing backbone (§2.35) adds a third
headroom-curve point (tree +37.2 over a collapsed flat 148.3). Remaining maze2d item:
the k290 composition control (§2.2). All maze2d confirmation is otherwise done.

## 2. Experiment queue (priority order, with decision rules)

### 2.1 DONE (2026-07-07) — DF seed replication: seeds 0–4, pooled t=3.90, +9.04. [commands kept for reference]

```bash
for i in 1 2 3 4; do python scripts/run_mcts_compare.py --env maze2d-large-v1 --method both --value-mode critic --df-ckpt final --budget 15 --k-mcts 16 --k-root 50 --top-m 3 --k-mcss 50 --n-envs 25 --n-episodes 1 --seed $i --out results/m2l_both_df_m3_s${i}.json; done
```

Then pooled paired analysis (works for any set of both-run files):

```bash
python - <<'EOF'
import json, glob, numpy as np
d = []
for f in sorted(glob.glob('results/m2l_both_df_m3_s*.json')):
    r = json.load(open(f))['results']
    d += list(np.array(r['mcts']['dv_norm'], float) - np.array(r['mcss']['dv_norm'], float))
d = np.array(d)
t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
print(f'pooled tree-MCSS diff {d.mean():+.2f}  n={len(d)}  paired t={t:.2f}')
EOF
```

**Decision rule:** pooled t ≥ 2.0 (n=125, 5 seeds) → the DF tree win is real; update
`notes/methodology_report.md` §7.5/§8 and `notes/value_lever_findings.md` §5d from
"preliminary" to confirmed. 1.0 < t < 2.0 → run seeds 5–9 (power ~doubles). t < 1.0 →
report as a suggestive null; the paper pivots to the triptych + kitchen (§4 below
still stands either way).

### 2.2 REQUIRED — compute-matched width control (~2.5 h)

The tree consumes 50 + 15×16 = 290 planner samples/step. Flat MCSS at the same budget:

```bash
python scripts/run_mcts_compare.py --env maze2d-large-v1 --method mcss --df-ckpt final --k-mcss 290 --n-envs 25 --n-episodes 1 --seed 0 --out results/m2l_mcss_df_k290_s0.json
```

**Decision rule** (compare to DF-tree-m3 192.2 and DF-MCSS-k50 182.7, same seed):
- k290 ≈ 190–193 → tree = compute-efficient sampling. Honest, still publishable as
  part of the law; the tree's sequential structure just concentrates samples well.
- k290 ≈ 183–186 → **tree = genuine composition** (search reaches plans no single
  sample contains, while flat width saturates). This is the paper-grade outcome.
- Run seeds 1–2 of whichever outcome occurs before writing it up.

### 2.3 OPTIONAL maze2d arms (only after 2.1–2.2)

- **Chunk expansion** (Frank's `chunk_size`; one flag, untested): add
  `--child-index 3` to the 2.1 command (keep budget 15 ⇒ 15×3=45 ≤ H−2=30 FAILS —
  use `--budget 9` with L=3: 27 ≤ 30 OK). Question: do segment-level branches beat
  waypoint-level?
- **Backbone-strength instrument** (tests the headroom prediction: better backbone ⇒
  smaller tree gain): `python scripts/train_df_planner.py --env maze2d-large-v1 --K 50 --steps 800000 --out-tag k50` (~7 h), then `scripts/check_df_ckpt.py --tag k50`,
  then repeat 2.1 with `--df-ckpt k50`. If DF-MCSS rises and the tree gap shrinks,
  the headroom clause of the law is confirmed; if the gap survives, composition is
  doing the work. Either outcome is a figure.

### 2.35 SPEED — cheap sampling gates + the shortcut-forcing planner (built 2026-07-06)

DF tree arms are slow (~2 h) because the pyramid runs 52 sweeps, dominated by the
causal-lag term (T−1=31), NOT the 20 noise levels. Two speed paths, in order:

**(a) Free gates on the EXISTING checkpoint (no retraining) — measure first:**
```bash
python scripts/check_df_ckpt.py --env maze2d-large-v1 --tag final --row-stride 2
python scripts/check_df_ckpt.py --env maze2d-large-v1 --tag final --row-stride 3
python scripts/check_df_ckpt.py --env maze2d-large-v1 --tag final --schedule fullseq
```
Compare gen-hop and DV-critic rows against the row_stride 1 pyramid reference
(hop 0.1586, critic −0.0025). If a cheap setting matches: closed-loop arms may use
`--df-row-stride N` for 2–3× speedup. **RULE: never change sampler settings within a
seed-pooled arm set** — the pending §2.1/§2.2 confirmation runs stay at row_stride 1;
cheap settings are for NEW arm sets and kitchen (re-run a seed-0 reference first).

**(b) Shortcut-forcing planner (Dreamer-4 recipe: Diffusion Forcing + Shortcut
Models [Frans et al., 2024, arXiv:2410.12557]) — built, untrained:**
`mcts/shortcut_df.py` (velocity param, dyadic d-conditioning, EMA self-consistency
bootstrap, per-token t) + `--shortcut` in train_df_planner.py. Samples in `sweeps`
(default 4) net forwards vs 52 → ~13× faster tree expansion; free tokens step
JOINTLY (no pyramid diagonal — gate (a)'s fullseq result predicts the quality cost).
```bash
python scripts/train_df_planner.py --env maze2d-large-v1 --shortcut --out-tag shortcut --smoke
python scripts/train_df_planner.py --env maze2d-large-v1 --shortcut --out-tag shortcut   # ~4 h
python scripts/check_df_ckpt.py --env maze2d-large-v1 --tag shortcut            # gates
python scripts/check_df_ckpt.py --env maze2d-large-v1 --tag shortcut --sweeps 8 # sweep count
# closed-loop gate, then arms — identical flags, loader auto-dispatches on cfg kind:
python scripts/run_mcts_compare.py --env maze2d-large-v1 --method mcss --df-ckpt shortcut --k-mcss 50 --n-envs 25 --n-episodes 1 --seed 0 --out results/m2l_mcss_dfshort_s0.json
```
**A shortcut planner is a DIFFERENT backbone** (collate tags it DFshor): never pool
with df_planner_ckpt_final arms; it needs its own seed-0 reference for every arm.
Intended use: kitchen-scale experiment volume, not the maze2d confirmation.
Gates same as DF: smoke → check (hops/critic/prefix) → MCSS closed-loop → tree.

### 2.4 Antmaze — do nothing

Backbone-limited (44% vs 76.9%) AND locomotion-capped (established: the ceiling is
the Ant toppling, not planning). It is a calibration row in the table. Do not retrain,
do not tree-search it, do not "fix" it. If a DF tree run is ever demanded there, add
`--junction-filter` (prunes the teleport tail).

### 2.5 THE MAIN ARC — kitchen (window 128/280: the env where depth has information)

1. **DV kitchen training — DONE (2026-07-07): `kitchen-mixed-v0`, planner+critic+policy
   @ 1M steps** (the sub-optimal `mixed` split — the headroom env the DV paper's own
   MCSS-lags-CFG-on-kitchen observation points to). Harness scoring FIXED same day:
   `run_episodes` now clips kitchen `dv_acc` to **[0,4]** per `veteran_d4rl_kitchen.py:447`
   (was mis-clipped to [0,1], which would cap a 4-subtask episode at 1). Still missing:
   kitchen V(s), kitchen DF planner, and any eval run.
2. **Reproduce DV-MCSS baseline** on kitchen — DONE 2026-07-07, reproduces at **75.0 ≈
   paper 73.6** (kitchen-mixed). Config: `--k-mcss 150 --critic-step 200000 --value-mode
   critic` (planner_temp 1.0, policy_temp 0.5, ddim/20 + ddpm/10 all already matched).
   **THE REAL BUG was `rebase_policy`, NOT K/critic:** the harness hardcoded rebase=True
   (right for maze2d/antmaze); kitchen's config is FALSE — dims 0-1 are joint angles, not
   xy, so rebasing corrupted the invdyn input. Rebased runs gave 60-64; un-rebased = 75.0.
   FIXED per-family in `mcts/specs.py` (`rebase_policy`: maze2d/antmaze True, kitchen
   False) + `--rebase-policy {0,1}` override in run_mcts_compare (prints the resolved
   value at startup). Maze2d/antmaze results UNAFFECTED (they were already True). Gotcha
   class: harness defaults vs the DV per-env config — check `configs/veteran/<fam>/*.yaml`.
   NOTE: MCSS dv_norm SEM ≈ 0 — every rollout completes ~exactly 3 of 4 subtasks (75.0).
   So the tree's job is to convert 3→4 on some fraction; if it can't, the next diagnostic
   is whether the DV planner even SAMPLES 4-task plans (the kitchen analog of the maze2d
   critic-ceiling check) — i.e. is the cap in selection (tree can help) or execution (cannot).
3. **Train kitchen V(s)** on subtask-return (`scripts/train_state_value.py`) — the
   pivot's original hypothesis: kitchen's value target is well-posed BY CONSTRUCTION.
   Keep the plan-value distillation recipe (`gen_plan_value_labels.py` +
   `train_plan_value.py`) as the fallback leaf value — it transferred to 5/5 envs.
4. **DV-backbone tree, day-one config:** `--value-mode critic` (composed windows) +
   `--top-m 3` + `--k-root 50`. Guard: budget×child_index ≤ H−1 (=31).
5. **DF trainer for kitchen — WIRED 2026-07-07** (gate deleted in train_df_planner.py;
   verified DV_D4RLKitchenSeqDataset.__getitem__ gathers seq_obs[p, start:start+(H-1)*
   stride+1:stride] = byte-identical to the DF sample_batch gather). CONTEXT that makes
   the DF run matter: **DV kitchen-mixed SATURATES at 75.0 (≈3 of 4 tasks)** — MCSS width
   scan stays flat at high K, DV-tree is null (74.0 vs 74.5, depth 2) — exactly like
   maze2d/antmaze on the DV backbone (planner pool has no 4-task plans for search to find).
   BUT the 4th task IS executable (DV kitchen-partial = 94 ≈ 3.76 tasks), so on *mixed* the
   cap is PLANNING/SAMPLING, not execution. So the DF test is sharp: does DF's diverse
   causal sampling surface 4-task plans the coherent full-sequence DV planner never does?
   KEY DIAGNOSTIC before the (expensive) tree — **DF-MCSS at high K: if it exceeds 75, the
   tree has a real shot at the 4th task**; if DF also saturates ≤75, kitchen is
   planner-capped regardless of backbone (honest boundary of the law).
   RESULT 2026-07-08: DF kitchen backbone trained (400k, loss ~3.0 = HEALTHY for D=60 —
   the eps-loss SUMS over dims so predict-zero ≈ 60, i.e. 3.0 = 0.05/dim, better-fit than
   maze2d's 0.09/dim). **Gate EXCELLENT: gen hop 0.259≈real 0.259, DV-critic gen
   0.2302≈real 0.2306 (gap 0.0004 vs maze2d's 0.065), hist_err 0.** DF-MCSS **60.5** (k150)
   / **57.0** (k600) — BELOW DV 75, width FLAT (k600≯k150, within noise) — same as DV
   (whose width also saturates at 75). So NEITHER backbone surfaces 4-task plans by
   sampling; the tree's **composition** (stitching a 4-task plan from cross-sample
   segments — what flat sampling can't do, and what kitchen's sub-horizon window is
   designed for) is the ONLY untested path. Backbone is sound → the DF-tree run is
   legitimate (next). Expectation: generality (tree helps DF, 2nd env) likely; beating DV
   needs composition to exceed maze2d's +9 by a lot. Cost: ~hours/run (52 DF sweeps) →
   shortcut backbone for scaling if the first tree looks promising.
   DF-TREE RESULT 2026-07-08 (n=25 s0, paired both-run): DF-tree **68.0** vs DF-MCSS
   **60.0** = **+8.0** (tree_depth 2, max 3) — the tree HELPS DF on kitchen: a 2nd env for
   the generality claim, mirroring maze2d's +9. BUT 68 < DV-MCSS 75 → the tree recovers
   ~half the DF-vs-DV gap (same as maze2d), NO DV-beating win; composition did NOT unlock
   the 4th task. Why: tree is SHALLOW (depth 2) and the generic DV critic is SATURATED on
   kitchen (shape-realism, gen-vs-real gap 0.0004) — it can't rank plans by task completion,
   so it can't steer search toward the rare 4-task plan. Next levers: (a) replicate 2-3
   seeds to firm +8 [do regardless]; (b) DEEPER tree (k_root 50 + budget 30-45, depth
   carries subtask info on kitchen unlike maze2d); (c) GROUNDED subtask reward on imagined
   states = a value that IMPROVES with depth (§2.5.6a) = the real shot at the 4th task the
   shape-critic can't see.
   SEED REPLICATION DONE 2026-07-08 (kitchen_both_df_s{0..3}.json, all configs verified:
   backbone=df, ckpt=final, k_root=150, top_m=3, critic 200k, seeds 0-3 distinct):
   per-seed tree-MCSS = +8, +10, +17, +7 (positive ALL seeds); POOLED n=100 **+10.5,
   paired t=5.47** — the kitchen DF-tree gain is CONFIRMED (bar was t>=2), stronger than
   maze2d's +9.04/t=3.90. Per-rollout diffs are quantized in 25-pt subtask units:
   tree completes +1 subtask in 43/100 rollouts (+2 in 5), -1 in 9, -2 in 1, tie in 42
   (sign test 48 wins vs 10 losses, p~4e-7). Means: DF-tree 70.0 vs DF-MCSS 59.5.
   STILL BELOW DV-MCSS 75.0 (best seed s2 hit 73.0) -> generality result banked, NOT a
   DV-beating win; and DV-tree is NULL on kitchen (74.0 vs 74.5, kitchen_both_tree_s0)
   -> the maze2d backbone-dichotomy REPLICATES on kitchen: tree helps DF (+10.5, t=5.5),
   tree does nothing for DV (-0.5, t=-0.6). Two envs x two backbones, one law.
   BONUS — kitchen already HAS its compute-matched control (the k290 analog maze2d still
   lacks): DF-MCSS k600 = 57.0 uses MORE planner windows than the tree (600 vs
   ~150+15x16=390) yet scores 13 BELOW it -> the kitchen tree gain is COMPOSITION/
   STRUCTURE, not sampling efficiency. Cite k600 as the width control in the writeup.
   Remaining paths to an absolute win stay (b) deeper tree and (c) grounded subtask
   reward — (c) is the principled one (the shape-saturated critic can't see task
   completion; a value that improves with depth is what search needs). DF training is
   reward-free (obs windows only), so kitchen's different value signature is
   irrelevant to it. Train (`--steps 400000`), gate with `scripts/check_df_ckpt.py`
   (hops + critic ballpark + prefix hist_err≈0), then DF-MCSS vs DV-MCSS, then the
   paired DF-tree vs DF-MCSS both-run — same ladder as maze2d.
6. **The novel kitchen designs** (in rising order of ambition; each is a contribution
   if it works): (a) grounded subtask reward computed on imagined states as the node
   value (evaluation that IMPROVES with depth — the MuZero ingredient nav never had);
   (b) terminus-node ("jumpy") expansion — branch on the END of each hallucinated
   window, natural at subtask boundaries (cf. MCTD, arXiv:2502.07202); (c)
   beyond-window DF rollout — `DFPlanner.sample` accepts any T > H (flexible
   horizon); score the first H with the critic, plan past it. (a) needs a kitchen
   subtask-completion checker on states; (b) needs a new expansion mode in
   `mcts_loop.py` (clone the `native` branch, child = window[-1], prefix reset);
   (c) is a sampling-call change only.
7. **GUIDANCE LEVERS (queued 2026-07-08, methodology_report §8.5 — all target the
   evaluator, which the 2x2 result showed is the binding constraint):**
   (d) **CFG-DF**: current DF net is UNCONDITIONAL (net(x,k) only, df_model.py) →
   CFG requires RETRAINING with a return-condition embedding + condition-dropout,
   then sample with w_cfg blend. Kitchen dataset already has the return labels
   (DV's own CFG path uses them, veteran_d4rl_kitchen.py guidance_type=cfg with
   center_mapping=False). Rationale: DV paper found CFG > MCSS on kitchen
   specifically (sub-optimal data); DF paper abstract itself advertises guiding
   schemes native to the causal architecture.
   (e) **CG on frozen DF via per-token noise-aware value V(x, k-vector)** — Frank's
   V(s, noise_lvl) idea and classifier guidance are THE SAME BUILD: train a critic
   on windows noised with per-token levels (the DF property applied to the value),
   steer sampling with its gradient; no planner retraining (the plug-and-play
   advantage Frank cited). Classical CG (Diffuser) is trajectory-level noise-aware;
   the PER-TOKEN version is the novel bit. Novelty check 2026-07-08:
   arXiv:2405.20555 = Diffusion Actor-Critic (Q-guidance inside policy TRAINING,
   no noise-conditioned value model) — does NOT clash. Same model doubles as an
   in-tree leaf evaluator for partially-denoised expansions.
   **BUILT 2026-07-11 (torch-free tests green locally; GPU smoke pending):**
   - mcts/noise_critic.py — NoiseAwareCritic: bidirectional DF-DiT (reuses
     CausalDFBlock, mask=None) + per-token noise adaLN + mean-pool scalar head
     (zero-init: guidance starts null). Labels = seq_val[path, start] in [-1,1]
     (same target the DV critic ranks). cfg kind="noise_critic".
   - mcts/df_schedule.py sample_training_levels — training k-mix: i.i.d.
     uniform (coverage) + pyramid rows with clean-history prefixes (the EXACT
     inference query distribution). Torch-free; tests/test_noise_critic_sched.py.
   - mcts/df_model.py DFPlanner.sample(guide=, w_cg=) — eps-shift CG:
     eps -= w*sqrt(1-ab[k])*grad_x V(x,k), history cols zeroed, per-token
     self-annealing; guide=None keeps all existing arms bit-identical.
   - scripts/train_noise_critic.py — path-level train/val split, dual eval
     (clean_corr @ k=0, sched_corr on ACTUALLY-NOISED sched-pattern inputs —
     review caught agent bug: clean-x + noisy-k eval was meaningless), saves
     noise_critic_ckpt_best.pt on best sched_corr (V(s)-style overfit guard).
     --smoke writes _smoke tag (never clobbers). --K must match DF K (20).
   - Wiring: run_mcts_compare --cg-ckpt/--cg-w (payload records both);
     check_df_ckpt --cg-ckpt/--cg-w (guided gate + noise-critic real-vs-gen
     score; REFUSES shortcut planner + K mismatch); Sampler guards (df-only,
     ckpt present, no shortcut, K match); collate tags DFcg<w> = never pools.
   EXECUTION LADDER (kitchen): (1) train smoke then full (~200k steps, watch
   sched_corr; loss should fall well below the val-label variance);
   (2) FREE guidance-strength sweep: check_df_ckpt --tag final --cg-ckpt best
   --cg-w {0.5,1,2,4} — pick largest w where hop p99 stays ~1.06-1.09 (real)
   while noise-critic gen score rises; (3) closed-loop DF-MCSS+CG at that w,
   n=25 (out: kitchen_mcss_df_cg<w>_s0.json) vs DF-MCSS 60.5 — if CG lifts
   the FLAT baseline toward/past 75, outcome 1 is live again; (4) tree on top.
   SWEEP RESULT 2026-07-11 (kitchen, DF final + noise-critic best): CLEAN
   MONOTONE PASS with no physical cost anywhere in range — hop p99 stays
   1.04-1.11 (real 1.09) and seam/cont-hop unchanged at every w, while scores
   rise near-linearly in w on BOTH evaluators: DV-critic gen (independent
   crosscheck) 0.2302 -> 0.2374 (w1) -> 0.2487 (w2) -> 0.2624 (w4) vs real
   0.2306, i.e. w4 generations score ABOVE real data; noise-critic gen
   0.2329 -> 0.2689. Ceiling NOT found at w=4 — gate w=8 (free) before/while
   running closed-loop at w=4. GOTCHA fixed same day: best-ckpt save now
   respects the _smoke guard (was clobber-able by a smoke run); re-sync
   scripts/train_noise_critic.py.
   TRAIN + w8 + CLOSED-LOOP RESULTS 2026-07-11:
   - noise critic trained 200k, BEST sched_corr = 0.915 @ 80k (overfit pattern
     as predicted -> deploy ckpt 'best', never 'final').
   - w=8 gate STILL CLEAN (hop p99 1.03 < real 1.09, seam 0.308, cont-hop
     1.02): DV-critic gen 0.2942 vs real 0.2306 (+0.064 above real). Ceiling
     still not found; do NOT sweep further until closed loop prices w8.
   - CLOSED LOOP cg_w=4 (kitchen_mcss_df_cg4_s0.json, n=25 s0): DF-MCSS+CG =
     64.0 +/- 2.5 vs unguided DF-MCSS 60.0 +/- 3.2 (matched n=25 s0 protocol;
     n=50 ref 60.5 +/- 2.0) = +3.5..4.0, DIRECTIONAL not yet significant
     (~1.3 SEM). Generation-side guidance closes ~1/4 of the DF->DV gap on
     the FLAT baseline alone. Guidance costs ~1.9x wall (2748s vs 1440s
     unguided: critic fwd+bwd per sweep).
   NEXT (in order): (1) THE MONEY RUN — guided tree at validated w4, paired:
   run_mcts_compare --env kitchen-mixed-v0 --method both --value-mode critic
   --df-ckpt final --cg-ckpt best --cg-w 4 --k-mcss 150 --critic-step 200000
   --k-root 150 --top-m 3 --budget 15 --k-mcts 16 --n-envs 25 --n-episodes 1
   --seed 0 --out results/kitchen_both_df_cg4_s0.json (~5-6h; unguided tree
   was +8..10 over unguided MCSS -> if effects compose, 64+8 ~ 72-75 =
   DV-MCSS parity/pass is IN RANGE); (2) dose-response: mcss-only cg_w=8
   (kitchen_mcss_df_cg8_s0.json, ~45min) — if w8 > w4 closed-loop, rerun the
   tree at w8; (3) if guided tree >= 75: seeds 1-3 = the headline win claim.
   MONEY-RUN + DOSE RESULTS 2026-07-11 (kitchen_both_df_cg4_s0.json,
   kitchen_mcss_df_cg8_s0.json): guided-w4 MCSS 64.0 / tree 70.0 (+6.0
   paired); w8 MCSS 66.0. Flat baseline dose-responds 60 -> 64 (w4) -> 66
   (w8), but the tree LANDS AT 70.0 = exactly the unguided tree's pooled
   landing point; the tree gain SHRANK 8 -> 6 under guidance. THIRD instance
   of the headroom law (after DV/DF/shortcut on maze2d): guidance lifts the
   flat pool, the tree equalizes to the same ~70 landing point. w8 both-run
   in flight — expect flat ~66 / tree ~70 (confirms the pin).
   W8 BOTH-RUN 2026-07-11: flat 66.0 / tree 70.0 (+4.0) — PIN CONFIRMED,
   prediction exact. The within-kitchen headroom curve is COMPLETE:
   flat 60/64/66 (w0/w4/w8) -> tree 68/70/70, gain +8/+6/+4 monotone-
   shrinking. The law's cleanest single-env figure; kitchen experiments DONE.
   FILE GOTCHA: the w8 both-run was saved over kitchen_both_df_cg4_s0.json
   (--out not changed). The JSON's internal cg_w field is authoritative
   (collate tags from it, not the filename) — verify cg_w==8.0 then
   mv to kitchen_both_df_cg8_s0.json. The w4 both-run's raw vectors are
   LOST (summary 64.0+/-2.5 / 70.0+/-2.0, +6.0 paired, recorded here);
   re-run only if paired w4 vectors are ever needed (~2.7h, low priority).
   *** THE 4-TASK DIAGNOSTIC (2026-07-11, decisive) ***: across ALL 750
   kitchen rollouts on disk (DV-MCSS 200/200 at exactly 75, DV width scans,
   DV-tree, DF-MCSS k150/k600, DF-tree s0-3, guided arms) NOT ONE rollout
   ever scored 100 — the 4th subtask has NEVER completed for ANY method.
   kitchen-mixed's defining property: demonstrations never solve all 4 tasks
   -> every LEARNED component (planner, invdyn policy, DV critic, noise
   critic) has labels/targets capped at the 3-task ceiling; a learned value
   CANNOT prefer a 4-task plan (out of label range). => DV-MCSS 75.0 is a
   HARD WALL for learned-value search on this split, not a selection gap.
   VERIFY the dataset claim (free, /workspace) — per-trajectory completed-
   subtask totals in the RAW data (rewards are +1-per-subtask events; the
   DV pipeline's [0,4] cumsum clip relies on exactly this):
   python - <<'EOF'
   import gym, d4rl, numpy as np
   env = gym.make('kitchen-mixed-v0'); d = env.get_dataset()
   r = d['rewards'].astype(float)
   done = d['terminals'].astype(bool) | d['timeouts'].astype(bool)
   tot, st = [], 0
   for i in np.where(done)[0]:
       tot.append(r[st:i + 1].sum()); st = i + 1
   u, c = np.unique(np.round(tot).astype(int), return_counts=True)
   print('per-trajectory subtask totals:', dict(zip(u.tolist(), c.tolist())))
   print('MAX =', max(tot), '(< 4 = no demo solves all 4 -> hard wall confirmed)')
   EOF
   CONSEQUENCES: (a) STOP chasing >75 via w-sweeps/deeper trees/backbones —
   physically cannot pay off; (b) the one lever that can EXPRESS the 4th
   task is the GROUNDED subtask checker (computed from state, not learned —
   not label-capped), §2.5.6a — but even then the policy must execute a
   never-demonstrated 4th-task context: time-boxed moonshot, not a default;
   (c) the kitchen chapter's claim set is COMPLETE as relative results:
   tree +10.5 (t=5.47, confirmed), guidance +4..6 (directional, n=25),
   guidance+tree compose to the equalizer point, boundary = dataset ceiling.
   (d) CHEAP significance for the CG flat claim if wanted: 2-3 mcss-only
   guided seeds (~46min each), NOT tree seeds (~2h each at 1.9x).
   (g) **DV-CG — the ORIGINAL trajectory-level classifier guidance, as a
   PRE-REGISTERED ceiling test (built 2026-07-11, not yet run).** Rationale:
   the DV pipeline's cg path trains the planner IDENTICALLY to MCSS
   (unconditional; only the side-model differs), so the existing planner +
   policy ckpts drop in — only the CumRewClassifier needs training.
   PREDICTION (registered before running): DV-CG <= 75.0 on kitchen-mixed —
   it optimizes the same label-capped returns as every learned value. If it
   lands > 75, the demonstration-ceiling claim is FALSIFIED (that risk is
   the point of the run). Bonus: completes the guidance-resolution
   comparison (trajectory-level CG on DV vs token-level CG on DF) and adds
   1000 census rollouts (the pipeline now prints the per-rollout
   distribution — watch for any 100).
   LADDER: (1) python scripts/train_dv_classifier.py --smoke  then full
   (~2-4h; builds the _cg dir, copies planner/policy ckpts, trains the
   classifier via the SDE's own update_classifier, aliases BEST ->
   classifier_ckpt_1000000.pt = what cg inference loads).
   (2) w-scan — the shipped task.planner_w_cfg=1.0 is a PLACEHOLDER (config's
   own comment: "Require grid tuning for CG"), 100 rollouts each (~40min):
   python pipelines/veteran_d4rl_kitchen.py mode=inference guidance_type=cg
   enable_wandb=false num_episodes=2 task.planner_w_cfg={0.5,1,2,4}
   (3) full protocol (1000 rollouts, ~6h) at the best w: same command,
   drop num_episodes override. Compare to DV-MCSS 75.0 / paper 73.6.
   NOTE: cg inference selects among 150 candidates by the classifier's OWN
   log_p (guided generation + classifier-ranked selection, no DV critic).
   CEILING VERIFICATION now a repo script (no heredoc/CRLF risk):
   python scripts/check_kitchen_ceiling.py            (mixed; expect MAX<4)
   python scripts/check_kitchen_ceiling.py --env kitchen-partial-v0 (control:
   expect 4s present — partial is where DV scores 94, proving the 4th task
   is executable when demonstrated).
   *** CEILING VERIFIED 2026-07-11 (corrected script) ***: mixed per-traj
   MAX-solved {0:1, 1:35, 2:312, 3:265} -> NO demo ever reaches 4 (best mode
   3-of-4 in 43% of demos; DV extracts it in 100% of rollouts = sits exactly
   AT the ceiling). partial control: {1:87, 2:259, 3:248, 4:19} -> 3.1%
   reach 4 (DV partial 94.0 = ~76% rollout conversion to the rare best
   mode). The DV paper's own mixed-vs-partial contrast (73.6 vs 94.0) = the
   INTERVENTION form of the ceiling claim — cite it instead of training a
   partial stack. PARTIAL-V0 DECISION: do NOT run it (full stack retrain
   ~2 GPU-days for a predicted small/null replication at flat=94; the
   published contrast already provides the moved-ceiling evidence). The
   boundary is now verified data-side AND census-side; results_chapter §7
   updated with both distributions + the extraction-reliability framing.
   *** SCRIPT CORRECTED 2026-07-11 — RE-RUN REQUIRED ***: first version
   summed the dataset reward field assuming sparse +1 events; the field is
   DENSE (per-step COUNT of currently-solved goal subtasks — proven by
   per-traj sums in the hundreds + the loader's "max discounted return
   401.6"). Env-at-eval rewards ARE sparse events (hence the [0,4] cumsum
   clip works closed-loop) — dataset and env reward forms DIFFER. Correct
   statistic = per-trajectory MAX of the reward field (max simultaneously-
   solved). The first run's "99.8% solve all 4" line is an ARTIFACT (sums
   vs 4) — the demonstration-ceiling premise is UNVERIFIED until the fixed
   script is re-run. Both envs report 613 trajectories = same demo corpus,
   relabeled per split's goal set.
   DV-CG SCAN partial result 2026-07-11: w=2.0 -> 74.75 +/- 0.25,
   distribution {50:1, 75:99} (100 rollouts) = PARITY with DV-MCSS 75.0,
   ZERO 4-task completions (census now 850+, still empty 100-bin);
   prediction <=75 HOLDING. GOTCHA: pipeline inference writes NO output
   files (stdout only; wandb off) — the w=0.5 and w=1.0 numbers were lost
   to terminal scrollback and must be RE-RUN with redirection:
   python pipelines/veteran_d4rl_kitchen.py mode=inference guidance_type=cg
   enable_wandb=false num_episodes=2 task.planner_w_cfg=0.5
   > results/dv_cg_scan_w0.5.log 2>&1
   (then: tail -3 results/dv_cg_scan_w0.5.log; same for 1.0, optionally 4.0)
   (h) **GROUNDED SUBTASK CHECKER — BUILT 2026-07-11 (torch-free tests green;
   GPU runs pending). The moonshot lever: the ONE evaluator not label-capped.**
   - mcts/grounded.py: numpy core cumulative_solved_count (union over ALL
     window steps INCL row 0 — a window from a 3-done state showing the 4th
     element at goal scores 4) + KitchenGroundedChecker (reads TASK_ELEMENTS/
     OBS_ELEMENT_INDICES/OBS_ELEMENT_GOALS/BONUS_THRESH off the LIVE env —
     nothing hardcoded; unnormalizes via the GaussianNormalizer; score() maps
     0..4 -> [-1,1]). tests/test_grounded.py 10/10 (torch-free).
   - mcts_loop.py: value_mode="grounded" = critic-mode plumbing with scorer
     _window_value = grounded.score + grounded_blend*critic (blend default
     0.25 = tiebreaker; 0 = pure grounded); --grounded-mcss reranks flat MCSS
     candidates the same way. Kitchen-only guards; all defaults inert.
   - run_mcts_compare: --value-mode grounded / --grounded-blend /
     --grounded-mcss (payload records all); collate tags "Gnd" = never pools.
   - scripts/check_grounded_pool.py = THE GO/NO-GO (open-loop, ~minutes):
     conditions DF on real dataset states with exactly --min-solved (default
     3) grounded-solved subtasks and asks whether ANY of k sampled windows
     imagines the (min_solved+1)-th completion — pure generalization beyond
     data support. Zero across the board => the generative model is also
     data-capped => moonshot dies cheaply, record boundary, stop.
   GROUNDED LADDER: (1) python scripts/check_grounded_pool.py --env
   kitchen-mixed-v0 --tag final --min-solved 3 --n 64 --k 150   [go/no-go]
   (1b) same + --cg-ckpt best --cg-w 8 [does guidance help imagine it? the
   label-cap argument predicts NO — either answer is a finding]
   (1c) sanity variant --min-solved 2 [should show plenty of 3s — validates
   the diagnostic itself before trusting a null at min-solved 3]
   §2.5.7h-RESULTS (2026-07-11, all three ran after the from_env fix):
   - dataset window-start solved-count scan (n=136,950): {0:29087, 1:43964,
     2:50497, 3:13402} mean 1.35 — only 9.8% of dataset states are 3-done.
   - min-solved=3, unguided: 9600/9600 windows score EXACTLY 3 (per-state max
     3 for all 64 states). 0 reach 4. The 3-done manifold is a DEAD END for
     the generator: demos that reach 3 stop there, and the causal model
     reproduces exactly that.
   - min-solved=3, cg_w=8: IDENTICAL 0/9600 — guidance (learned, label-capped)
     cannot summon what its labels never contained; 3rd independent
     confirmation of the label-cap mechanism (after DV-CG 74.75 and the
     rollout census).
   - min-solved=2 (validation + surprise): [2:6869, 3:2707, 4:24] — 28.45% of
     windows imagine the DEMONSTRATED 2->3 transition (diagnostic validated),
     and 24 windows across 7/64 states imagine FOUR solved — ABOVE the
     demonstration ceiling. The generator is NOT strictly data-capped; it can
     compose 4-solved imaginations, but only from below the bottleneck (2-done
     contexts, where continuation diversity is rich), never from the dead-end
     3-done manifold. CAVEATS: 0.25% rate; union semantics counts transient
     near-goal passes, so some/all of the 24 may be unexecutable hallucinated
     grazes — exactly what the closed loop (grounded selection prefers them
     at 2-done steps, policy must execute) now tests.
   (2) IF GO: closed loop, paired:
   python scripts/run_mcts_compare.py --env kitchen-mixed-v0 --method both
   --value-mode grounded --grounded-mcss 1 --df-ckpt final --k-mcss 150
   --critic-step 200000 --k-root 150 --top-m 3 --budget 15 --k-mcts 16
   --n-envs 25 --n-episodes 1 --seed 0
   --out results/kitchen_both_df_grounded_s0.json
   WATCH: any per-rollout 100 = the demonstration wall falls (would be the
   headline). TIME-BOX: if the go/no-go is a hard zero, write it up as the
   boundary's final exhibit + future work; do NOT iterate on sampler tricks.
   (i) **CG-seeds + analyzer (item 3) — DONE 2026-07-11.** Seeds ran
   (kitchen_mcss_df_cg4_s1, cg8_s1, cg8_s2); scripts/analyze_kitchen_cg.py
   verdict: **w=8 CONFIRMED +4.67 paired t=2.22 n=75 (3 seeds, 24W/12L/39T);
   w=4 NOT separable +1.50 t=0.46 n=50** — the dissertation quotes the w=8
   claim only (results_chapter §6 + limitations updated accordingly).
   Pooled flat means: 59.8 (w0, n=150) / 62.0 (w4, n=50) / 64.3 (w8, n=100).
   Do NOT run more CG flat seeds — the claim is settled at both doses.
   FILE HYGIENE: local results/ still holds kitchen_both_df_cg4_s0.json with
   cg_w=8 INSIDE (the overwritten money-run file) alongside
   kitchen_both_df_cg8_s0.json — if they are copies of the same run, DELETE
   the cg4-named one (pooled counts would double-count; the analyzer dedupes
   pairing but not pooling).
   WRITE-UP STATUS 2026-07-11 (updated): notes/results_chapter.md = results
   chapter DRAFT (now incl. the VERIFIED ceiling distributions, the DV-CG
   fourth-family row, and the extraction-reliability framing: mixed best
   mode 3-of-4 in 43% of demos -> DV extracts 100%; partial 4-of-4 in 3.1%
   -> DV converts ~76%). notes/intro_chapter.md = introduction DRAFT
   (motivation/tension, RQ evolution, the 3-clause law, 6 contributions,
   scope framing, structure). notes/related_work_chapter.md = related-work
   DRAFT (6 positioning sections each ending "This work:", + positioning
   table; [VERIFY] flags on: Ho&Salimans venue, 2405.20555 authors, Dreamer4
   citation, Brandfonbrener 2022, Emmons RvS, Gupta 2019 relay). Remaining:
   discussion/conclusion chapter, the 6 figures from JSONs, [VERIFY] pass,
   DV-CG w-scan completion (w0.5/1.0 re-runs with redirection).
   (f) **Shortcut on kitchen — NOT YET TRAINED** (maze2d only). ~6x cheaper
   expansion = the enabler for the deeper-tree probe (budget 30-45) and for
   sweeping (d)/(e). Train: `python scripts/train_df_planner.py --env
   kitchen-mixed-v0 --shortcut --out-tag shortcut` then gate with check_df_ckpt
   --tag shortcut (watch hop p99 vs real 1.09 — the maze2d shortcut showed a hop
   tail); run tree arms with `--df-ckpt shortcut --sweeps 8`. Final paper numbers
   stay on full DF (52-sweep) per §7.6 protocol.

## 3. Reporting standards (used throughout — keep them)

- Paired designs everywhere: `--method both` shares envs; separate runs with the same
  `--seed` also share envs (verify `starts` arrays match before pairing across files).
- Report per-arm mean ± SEM AND the paired t on per-env differences; 3+ seeds pooled
  before any claim; McNemar via `python scripts/collate_mcts.py` for reach%.
- Collate labels: DF arms carry the `DF` suffix, inpaint `inp`, plan-value `Vplanv`,
  top-m `m3`, wide root `r50` — never pool differently-labeled arms.

## 4. The paper (what it is, regardless of remaining outcomes)

**Working title:** "When Does Tree Search Help Diffusion Planners?"
**Core contribution — a law with each clause measured:** search helps iff
(1) node values are well-posed [V(s) 0.74→0.98 corr, +35 closed-loop, 5 envs];
(2) nodes are scored on comparable windows [composed-window fix, ~1.5];
(3) backups are noise-robust [winner's curse −5.7, top-m +4.5 p<1e-4];
(4) expansion is faithful conditional generation [the triptych: glue −6.8 /
inpaint −22.8 / DF +8..9 — monotone in expansion fidelity, with the "seam was the
critic's accidental defense" diagnosis];
(5) the flat baseline leaves evaluator headroom [DV saturates at ~206; DF does not].
**Figures:** the triptych bar chart; the posedness table (5 envs); depth-score
inflation profiles (diag_inpaint); width-saturation vs tree curves (incl. k290);
the causal-ladder table (167→202.5→203→205→206).
**Positioning:** MCTD (ICML'25) is the concurrent positive half; this paper supplies
the negative regime + the mechanism + the conditions. Venue ladder: NeurIPS/ICLR
workshop (safe now) → ICLR/ICML main track (needs kitchen arm + confirmed DF result).
**Dissertation:** `notes/methodology_report.md` §2–§7 are the methodology/results
chapters; §10 is the conclusion skeleton. Verify all citations (flagged inside).

## 5. Do-NOT-reopen list (measured dead ends — cite them, don't re-run them)

- A "win over MCSS" on maze2d/antmaze with the DV backbone (ceiling ~206 measured).
- Behaviour-return V(s) (ill-posed, falsified); oracle geodesic values (dynamics-blind).
- Critic fine-tuning on stitched windows (Lever A: aleatoric floor, damages baseline).
- Inpaint expansion as a *fix* (it is the measured −16 ablation arm; keep it in tables).
- Antmaze polish of any kind (locomotion cap).
- Loss-value comparisons across objectives (DF 0.36 vs DV 0.04 is expected — §7.3 of
  the methodology report explains; do not "debug" it).

## 6. File map

- `notes/methodology_report.md` — the science, algorithms, citations (VERIFY cites).
- `notes/value_lever_findings.md` — value-lever arc detail (§5b inpaint, §5d DF).
- `notes/findings_summary.md`, `notes/PROJECT_HANDOFF.md` — older arcs (handoff stale past §7).
- `mcts/df_model.py`, `mcts/df_schedule.py`, `scripts/train_df_planner.py`,
  `scripts/check_df_ckpt.py` — the DF stack. `mcts/window.py` — composed windows +
  inpaint prior. `mcts/stitch.py` — exact stitched labels. `scripts/run_mcts_compare.py`
  — the runner (all flags). `scripts/collate_mcts.py` — tables + McNemar.
  `scripts/diag_inpaint.py` — the inpaint diagnostic. Tests: `tests/` (torch-free,
  run `python -m pytest tests/test_df_schedule.py tests/test_inpaint_prior.py tests/test_compose_window.py tests/test_collate_mcts.py tests/test_topm_backup.py tests/test_value_forest.py tests/test_stitch.py -q` → 63 pass).
