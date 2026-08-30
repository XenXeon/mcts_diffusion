# scripts/

The live toolchain for the MCTS-into-Diffusion-Veteran study. Each script is a
standalone entry point run from the repo root (`python scripts/<name>.py`); it
inserts the repo root on `sys.path` and imports the `mcts` package. Shared
constants (checkpoint roots, H/stride, dataset target config) live in
`mcts/specs.py` — import them, do not re-declare. One-off scripts from the
closed Phase 0–6 investigations are archived in `legacy/` (see `legacy/README.md`).

## Live pipeline

**Training**
| Script | Role |
|---|---|
| `train_df_planner.py` | Diffusion-Forcing planner (`--shortcut` = few-step variant) |
| `train_noise_critic.py` | per-token noise-aware value V(x, k) for classifier guidance |
| `train_dv_classifier.py` | the DV pipeline's original trajectory-level CG classifier |
| `train_state_value.py` | learned state-value V(s) / V(s, g) |
| `gen_plan_value_labels.py` → `train_plan_value.py` | Lever B: distilled plan-value V̂(s) |

**Evaluation**
| Script | Role |
|---|---|
| `run_mcts_compare.py` | main harness: paired MCSS vs tree, any backbone/value/guidance; writes one JSON per cell with per-rollout vectors (pairing key = seed + index) |
| `collate_mcts.py` | pool cell JSONs into the candidates-vs-reach table + exact McNemar tests |
| `analyze_kitchen_cg.py` | pooled/paired analysis of the kitchen guidance arms |

**Diagnostics** (read-only, open-loop)
| Script | Role |
|---|---|
| `check_df_ckpt.py` | planner fidelity gate: hops, DV-critic ballpark, prefix conditioning |
| `check_grounded_pool.py` | grounded go/no-go — can the planner imagine the 4th subtask? |
| `check_kitchen_ceiling.py` | raw-data demonstration ceiling (max subtasks solved per demo) |

**Figures**
| Script | Role |
|---|---|
| `make_figures.py` | the six dissertation figures → `figures/` (PDF + PNG) |

Current commands with expected outputs and decision rules are in
`notes/NEXT_STEPS_RUNBOOK.md`; the methodology is `notes/methodology_report.md`.

## Corrected conclusions — do not re-trip on these

Institutional knowledge from the closed phases, kept here because it is easy to
rediscover the wrong way:

1. **A weak open-loop return correlation does not mean the critic is bad.** The
   Phase-1 `r = −0.116` was underpowered and range-restricted; the critic reads
   whole plans near-perfectly (r = −0.995 vs plan reach-time).
2. **Critic self-consistency is not task performance.** Deeper trees reaching
   critic ≈ 1.0 did not control better — the winner's curse, later quantified.
3. **Single-shot reach is not a proxy for closed-loop success.** DV replans every
   step; only full-episode score counts. The phase-5/6 headroom scripts are
   single-shot diagnostics — read them accordingly.
4. **The legacy tree engine (`mcts/tree|node|expansion|rollout`) uses MEAN backup
   of full-trajectory critic scores** — the structure that provably cannot benefit
   from depth. The current engine is `mcts/value_forest.py` (state-value, MAX/top-m
   backup); the closed-loop sampler is `mcts/mcts_loop.py`.
