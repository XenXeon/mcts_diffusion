# scripts/legacy/

One-off scripts from the closed Phase 0–6 investigations, kept for
reproducibility and provenance — **not part of the current experiment loop**.
Their results are recorded in `notes/report_phases_0_to_4.md`,
`notes/writeup_phases_0_to_4.md`, and `notes/findings_summary.md`. Most predate
`mcts/specs.py` and intentionally keep their own local constants (documenting
the exact configuration each recorded result used); they may lag later harness
changes. Run from the repo root if reproducing (`python scripts/legacy/<name>.py`).

| Phase / group | Scripts | Recorded in |
|---|---|---|
| 0–1 baseline & verification | `p1_preverification`, `p1_lengthcheck`, `phase1_ground_truth_returns`, `phase1_make_plots`, `test_start_state_generalisation`, `run_one_episode` | report §2–3 |
| 2 expansion primitive | `phase2_smoke_test` | §4 |
| 3 tree mechanics | `phase3_ablation`, `phase3_k_ablation` | §5 |
| 4 closed-loop MCTS v1 | `phase4_mcts_rollout`, `phase4_ablation` | §6 |
| 5 ceiling diagnosis | `phase5_headroom_diagnostic`, `phase5_plan_diversity`, `phase5_child_index_ablation`, `phase5_critic_depth_calibration`, `phase5_zero_return_diagnosis`, `inspect_zero_traces` | §7 |
| 6 oracle / headroom | `phase6_stage0_oracle`, `phase6_headroom_any` | `notes/phase6_muzero_design.md` |
| antmaze diagnostics | `diag_d1_compass`, `diag_d2_exploitability`, `diag_d4_calibration`, `diag_fall_geometry`, `diag_wall_blindness`, `diag_oracle_flat`, `diag_oracle_tree`, `diag_inpaint` | `notes/findings_summary.md` |
| closed value levers | `finetune_critic_stitched` (Lever A), `eval_state_value` (V(s) selector check) | `notes/value_lever_findings.md` |
| visualization / misc | `animate_compare`, `animate_failure`, `plot_candidates`, `plot_failures`, `analyze_failures`, `measure_dmax`, `run_compare_trace`, `run_instrumentation`, `make_report_figures` | — |

The current dissertation figures come from `scripts/make_figures.py` (not the
older `make_report_figures.py` archived here).
