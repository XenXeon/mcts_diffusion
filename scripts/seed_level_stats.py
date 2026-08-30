"""scripts/seed_level_stats.py

Recompute every load-bearing comparison in the dissertation with SEED-LEVEL
statistics as primary, per the protocol in Chapter 3.

Rationale: pooling per-rollout differences across seeds and treating them as
independent overstates the degrees of freedom, because rollouts within a seed
share a start draw and a checkpoint. The seed is the unit of replication. This
script reports, for every comparison:

    diff        mean paired difference (seed-level mean of per-seed means)
    seed-t      paired t over per-seed mean differences   <- PRIMARY
    roll-t      paired t over pooled per-rollout diffs     <- secondary
    CI95        95% confidence interval on the seed-level mean
    d           Cohen's d over the per-seed differences
    starts      whether the compared arms' `starts` arrays were asserted equal

Every comparison is start-matched and seed-matched: an arm is only differenced
against a baseline restricted to that arm's own seeds.

Run (torch-free):  python scripts/seed_level_stats.py [--md notes/seed_level_stats.md]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")

# t critical values for a two-sided 95% interval, df = n-1
TCRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
         7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def load(pattern, arm=None):
    """{seed: (dv_norm vector, starts array)} over files matching pattern."""
    out = {}
    for f in sorted(glob.glob(pattern)):
        j = json.load(open(f))
        r = j.get("results", {})
        a = r.get(arm) if arm else None
        if a is None:
            if arm and arm in r:
                a = r[arm]
            elif arm:
                continue
            else:
                a = list(r.values())[0]
        v = np.asarray(a.get("dv_norm") or [], float)
        if not len(v):
            continue
        out[j.get("seed", 0)] = (v, np.asarray(a.get("starts") or [], float))
    return out


def compare(label, A, B, note=""):
    seeds = sorted(set(A) & set(B))
    if not seeds:
        return dict(label=label, ok=False, note="no shared seeds")
    per_seed, pooled, starts_ok = [], [], True
    for s in seeds:
        a, sa = A[s]
        b, sb = B[s]
        n = min(len(a), len(b))
        if len(sa) and len(sb) and sa.shape == sb.shape:
            starts_ok &= bool(np.allclose(sa[:n], sb[:n]))
        d = a[:n] - b[:n]
        per_seed.append(d.mean())
        pooled.append(d)
    ps = np.asarray(per_seed)
    pl = np.concatenate(pooled)
    ns = len(ps)
    if ns > 1:
        sd = ps.std(ddof=1)
        sem = sd / np.sqrt(ns)
        st = ps.mean() / sem if sem > 0 else np.nan
        tc = TCRIT.get(ns - 1, 1.96)
        ci = (ps.mean() - tc * sem, ps.mean() + tc * sem)
        d_eff = ps.mean() / sd if sd > 0 else np.nan
    else:
        st, ci, d_eff = np.nan, (np.nan, np.nan), np.nan
    rt = pl.mean() / (pl.std(ddof=1) / np.sqrt(len(pl))) if len(pl) > 1 else np.nan
    return dict(label=label, ok=True, diff=ps.mean(), seed_t=st, roll_t=rt,
                ci=ci, d=d_eff, n_seeds=ns, n_roll=len(pl),
                starts=starts_ok, note=note)


def build():
    rows, sections = [], []

    def sec(name):
        sections.append((name, len(rows)))

    # ── maze2d, full-sequence backbone (Chapter 4) ────────────────────────
    sec("maze2d-large, full-sequence backbone (Chapter 4)")
    k50 = load("results/maze2d_large_mcss_k50_s*.json", "mcss")
    k256 = load("results/maze2d_large_mcss_k256_s*.json", "mcss")
    naive = load("results/maze2d_large_critic_tree_s*.json", "mcts")
    naive[0] = load("results/maze2d_large_critic_tree.json", "mcts")[0]
    rows.append(compare("naive critic tree vs MCSS k50", naive, k50, "10 seeds"))
    mx = load("results/m2l_tree_r50_s*.json", "mcts")
    t3 = load("results/m2l_tree_r50_m3_s*.json", "mcts")
    rows.append(compare("tree MAX backup vs MCSS k50", mx, k50))
    rows.append(compare("tree top-3 backup vs MCSS k50", t3, k50))
    rows.append(compare("tree top-3 backup vs MCSS k256", t3, k256, "compute-matched"))
    rows.append(compare("top-3 vs MAX (winner's curse fix)", t3, mx, "same starts"))
    naive_vs = load("results/m2l_tree_vsnaive_s*.json", "mcts")
    planv = load("results/m2l_tree_planv_s*.json", "mcts")
    rows.append(compare("behaviour-return V(s) tree vs MCSS k50", naive_vs, k50))
    rows.append(compare("plan-value tree vs MCSS k50", planv, k50))
    rows.append(compare("plan-value - behaviour-return (posedness)", planv, naive_vs,
                        "same starts, same config; only the value checkpoint differs"))
    rows.append(compare("inpaint tree vs MCSS k50",
                        load("results/m2l_tree_criticr50_inpaint.json", "mcts"), k50,
                        "single seed"))
    rows.append(compare("MCSS k256 vs k50 (width control)", k256, k50, "10 seeds"))
    rows.append(compare("stitched-critic MCSS vs MCSS k50",
                        load("results/m2l_mcss_k50_stitched_s0.json", "mcss"), k50,
                        "Lever A"))

    # ── the value-posedness ladder (Chapter 4) ────────────────────────────
    sec("goal-conditioned value ladder (Chapter 4)")
    for env, tag, base in [("maze2d-large", "m2l", "results/maze2d_large_mcss_k256_s*.json"),
                           ("maze2d-medium", "m2m", "results/m2m_mcss_k256_s*.json"),
                           ("maze2d-umaze", "m2u", "results/m2u_mcss_k256_s*.json")]:
        rows.append(compare(f"V(s,g)-pess tree vs k256 — {env}",
                            load(f"results/{tag}_tree_vsgpess_s*.json", "mcts"),
                            load(base, "mcss")))

    # ── faithful-conditioning backbones (Chapter 5) ───────────────────────
    sec("faithful-conditioning backbones (Chapter 5)")
    for lbl, pat in [("DF tree vs DF-MCSS — maze2d-large", "results/m2l_both_df_m3_s*.json"),
                     ("shortcut tree vs shortcut-MCSS — maze2d-large",
                      "results/m2l_both_dfshort8_m3_s*.json"),
                     ("DF tree vs DF-MCSS — kitchen", "results/kitchen_both_df_s*.json"),
                     ("DV tree vs DV-MCSS — kitchen", "results/kitchen_both_tree_s*.json")]:
        rows.append(compare(lbl, load(pat, "mcts"), load(pat, "mcss"), "within-file"))

    # ── guidance (Chapter 6) ──────────────────────────────────────────────
    sec("per-token noise-aware guidance (Chapter 6)")
    ung = load("results/kitchen_mcss_df_s*.json", "mcss")
    unguided_seeded = {}
    for f in sorted(glob.glob("results/kitchen_both_df_s*.json")):
        j = json.load(open(f))
        a = j["results"]["mcss"]
        unguided_seeded[j["seed"]] = (np.asarray(a["dv_norm"], float),
                                      np.asarray(a.get("starts") or [], float))
    def cg_flat(w):
        d = load(f"results/kitchen_both_df_cg{w}_s*.json", "mcss")
        for s, v in load(f"results/kitchen_mcss_df_cg{w}_s*.json", "mcss").items():
            d.setdefault(s, v)          # union; overlapping seeds are bit-identical
        return d
    rows.append(compare("flat +CG w=8 vs unguided", cg_flat(8), unguided_seeded))
    rows.append(compare("flat +CG w=4 vs unguided", cg_flat(4), unguided_seeded))
    rows.append(compare("guided tree vs guided flat, w=0 (the pin)",
                        load("results/kitchen_both_df_s*.json", "mcts"), unguided_seeded))
    for w in (4, 8):
        rows.append(compare(f"guided tree vs guided flat, w={w} (the pin)",
                            load(f"results/kitchen_both_df_cg{w}_s*.json", "mcts"),
                            cg_flat(w)))
    rows.append(compare("grounded tree vs grounded flat",
                        load("results/kitchen_both_df_grounded_s0.json", "mcts"),
                        load("results/kitchen_both_df_grounded_s0.json", "mcss"),
                        "single seed"))

    # ── denoising-axis port (Chapter 5) ───────────────────────────────────
    sec("denoising-axis search, MCTD port (Chapter 5)")
    mpc = load("results/mcssmpc_maze2d_large_rp50_s*.json")
    rows.append(compare("MCTD (as published) vs MCSS-MPC",
                        load("results/mctd_maze2d_large_rp50_s*.json"), mpc))
    rows.append(compare("MCTD-critic vs MCSS-MPC",
                        load("results/mctdcritic_maze2d_large_rp50_s*.json"), mpc))
    rows.append(compare("guided best-of-N vs MCSS-MPC",
                        load("results/guidedbon_maze2d_large_rp50_s*.json"), mpc))
    for env, tag in [("medium", "medium"), ("umaze", "umaze")]:
        rows.append(compare(f"MCTD-critic vs MCSS-MPC — maze2d-{env}",
                            load(f"results/mctdcritic_maze2d_{tag}_rp50_s*.json"),
                            load(f"results/mcssmpc_maze2d_{tag}_rp50_s*.json")))

    # ── execution-model factors (Chapter 5) ───────────────────────────────
    sec("execution-model factors (Chapter 5)")
    rp1 = load("results/mcssmpc_maze2d_large_cad_rp1_s*.json")
    rp10 = load("results/mcssmpc_maze2d_large_cad_rp10_s*.json")
    rp100 = load("results/mcssmpc_maze2d_large_cad_rp100_s*.json")
    rows.append(compare("cadence: rp10 vs rp1 (DF, one harness)", rp10, rp1))
    rows.append(compare("cadence: rp100 vs rp1 (DF, one harness)", rp100, rp1))
    rows.append(compare("backbone: DV vs DF at rp1 (one harness)",
                        load("results/mcssmpc_maze2d_large_cad_rp1_dv_s*.json"),
                        load("results/mcssmpc_maze2d_large_cad_rp1_df_s*.json")))
    rows.append(compare("backbone: DV vs DF at rp50 (one harness)",
                        load("results/dvmcssmpc_maze2d_large_rp50_s*.json"), mpc))
    return rows, sections


def fmt(rows, sections):
    idx = {i: n for n, i in sections}
    out = ["# Seed-level statistics for every load-bearing comparison", "",
           "*Generated by `scripts/seed_level_stats.py`. Seed-level paired t is the",
           "primary statistic (the seed is the unit of replication); the pooled",
           "per-rollout t is secondary and is shown for comparison. Every arm is",
           "differenced against a baseline restricted to its own seeds, and `starts`",
           "equality is asserted before differencing.*", "",
           "`d` is Cohen's d over the per-seed differences. A dash means a single",
           "seed, where no seed-level statistic is defined.", ""]
    for i, r in enumerate(rows):
        if i in idx:
            out += ["", f"## {idx[i]}", "",
                    "| comparison | diff | **seed-t** | roll-t | CI95 | d | seeds | n | starts |",
                    "|---|---|---|---|---|---|---|---|---|"]
        if not r.get("ok"):
            out.append(f"| {r['label']} | — | — | — | — | — | — | — | {r.get('note','')} |")
            continue
        st = "—" if np.isnan(r["seed_t"]) else f"**{r['seed_t']:+.2f}**"
        ci = "—" if np.isnan(r["ci"][0]) else f"[{r['ci'][0]:+.1f}, {r['ci'][1]:+.1f}]"
        de = "—" if np.isnan(r["d"]) else f"{r['d']:+.2f}"
        ok = "yes" if r["starts"] else "**NO**"
        out.append(f"| {r['label']} | {r['diff']:+.2f} | {st} | {r['roll_t']:+.2f} | "
                   f"{ci} | {de} | {r['n_seeds']} | {r['n_roll']} | {ok} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--md", default="notes/seed_level_stats.md")
    args = p.parse_args()
    rows, sections = build()
    text = fmt(rows, sections)
    os.makedirs(os.path.dirname(args.md), exist_ok=True)
    open(args.md, "w", encoding="utf-8").write(text)
    print(text)
    print(f"\nwrote {args.md}")
