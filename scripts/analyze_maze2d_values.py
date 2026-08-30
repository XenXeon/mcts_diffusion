"""scripts/analyze_maze2d_values.py

Pool + pair the maze2d value-mode arms for the final comparison table: DV-MCSS
(shared flat baseline), DV-tree critic, V(s) tree, V(s,g) tree, V(s,g)-pess
tree, plus the DF/shortcut arms. Reports seed-level mean+/-SEM and pooled
per-rollout mean+/-SEM, and — where a tree arm and a DV-MCSS run share a seed
(hence the same starts/goals) — a PAIRED per-rollout test, since --seed fixes
the env reset. Torch-free; reads results/m2l_*.json.

The V(s,g)-pess arm is the one under scrutiny (it edged above MCSS at 5 seeds):
this is the tool that decides whether that survives pairing + more seeds.

Run:  python scripts/analyze_maze2d_values.py
"""
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np


def load(f):
    try:
        return json.load(open(f))
    except Exception:
        return None


def classify(d):
    """Return (arm_label, method_arm) or None. method_arm is the results key to read."""
    bb = d.get("backbone")
    dfk = d.get("df_ckpt")
    sw = d.get("df_sweeps") or d.get("sweeps")
    vm = d.get("value_mode", "critic")
    r = d.get("results", {})
    has_mcts = bool(r.get("mcts", {}).get("dv_norm"))
    has_mcss = bool(r.get("mcss", {}).get("dv_norm"))
    if bb == "df" and (dfk == "shortcut" or (sw in (4, 8))):
        return ("shortcut tree" if has_mcts else "shortcut MCSS",
                "mcts" if has_mcts else "mcss")
    if bb == "df":
        return ("DF tree" if has_mcts else "DF MCSS",
                "mcts" if has_mcts else "mcss")
    # DV backbone (bb == 'dv' or None on older files)
    if has_mcts:
        lab = {"critic": "DV-tree critic", "v_s": "DV-tree V(s)",
               "v_sg": "DV-tree V(s,g)", "v_sg_pess": "DV-tree V(s,g)-pess"}.get(vm, f"DV-tree {vm}")
        return (lab, "mcts")
    if has_mcss:
        return ("DV-MCSS", "mcss")
    return None


def stats_seedlevel(per_seed_means):
    v = list(per_seed_means.values())
    n = len(v)
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
    return m, sd / math.sqrt(n) if n > 1 else 0.0, n


def main():
    # arm -> seed -> per-rollout vector. Glob ALL maze2d files by env (the k50
    # MCSS baseline is named maze2d_large_mcss_k50_s*, NOT m2l_*; globbing m2l_*
    # only would fall back to the weak k16 files — the baseline bug this fixes).
    arms = defaultdict(dict)
    starts = defaultdict(dict)     # arm -> seed -> starts array (for start-matched pairing)
    for f in sorted(glob.glob("results/*.json")):
        d = load(f)
        if not isinstance(d, dict) or "maze2d" not in str(d.get("env", "")):
            continue
        r = d.get("results")
        if not isinstance(r, dict):
            continue
        c = classify(d)
        if not c:
            continue
        lab, key = c
        # DV-MCSS is k-dependent — keep k50/k16/k256 as distinct arms so the
        # tree's flat baseline (k_root 50 => DV-MCSS k50) is never conflated.
        if lab == "DV-MCSS":
            lab = f"DV-MCSS k{d.get('k_mcss')}"
        v = [float(x) for x in r[key]["dv_norm"]]
        seed = d.get("seed")
        if seed not in arms[lab] or len(v) > len(arms[lab][seed]):
            arms[lab][seed] = v
            st = r[key].get("starts") or d.get("starts")
            starts[lab][seed] = np.asarray(st) if st is not None else None

    order = ["DV-MCSS k50", "DV-MCSS k256", "DV-MCSS k16", "DV-tree critic",
             "DV-tree V(s)", "DV-tree V(s,g)", "DV-tree V(s,g)-pess",
             "DF MCSS", "DF tree", "shortcut MCSS", "shortcut tree"]
    print(f"{'arm':22s} {'seeds':16s} {'seed mean±SEM':16s} {'pooled mean±SEM (n)'}")
    print("-" * 78)
    for lab in order + [a for a in arms if a not in order]:
        if lab not in arms:
            continue
        by_seed = arms[lab]
        seed_means = {s: float(np.mean(v)) for s, v in by_seed.items()}
        m, se, n = stats_seedlevel(seed_means)
        pooled = np.concatenate([np.asarray(v) for v in by_seed.values()])
        psem = pooled.std(ddof=1) / math.sqrt(len(pooled))
        sk = sorted(k for k in by_seed if k is not None)
        print(f"{lab:22s} {str(sk):16s} {m:6.2f} ± {se:4.2f}     "
              f"{pooled.mean():6.2f} ± {psem:4.2f} (n={len(pooled)})")

    # ── paired tests: each DV-tree arm vs BOTH DV-MCSS baselines, start-matched.
    # k50 = root-width baseline; k256 = the ~compute-matched baseline for the
    # tree's ~290 planner samples (the decisive comparison). Report seed-level t
    # (n=seeds, the "does it generalise across seeds" test) alongside per-rollout.
    for bname in ("DV-MCSS k50", "DV-MCSS k256"):
        base, bstarts = arms.get(bname, {}), starts.get(bname, {})
        if not base:
            print(f"\n(no {bname} arm found for pairing)")
            continue
        print(f"\nPAIRED vs {bname} (start-matched):")
        for lab in ["DV-tree critic", "DV-tree V(s)", "DV-tree V(s,g)",
                    "DV-tree V(s,g)-pess"]:
            if lab not in arms:
                continue
            diffs, seed_d, used, unmatched = [], [], [], 0
            for s, tv in arms[lab].items():
                bv = base.get(s)
                if bv is None or len(bv) != len(tv):
                    continue
                ts, ms = starts[lab].get(s), bstarts.get(s)
                if ts is not None and ms is not None and not np.allclose(ts, ms):
                    unmatched += 1
                    continue
                diffs += [a - b for a, b in zip(tv, bv)]
                seed_d.append(float(np.mean(tv) - np.mean(bv)))
                used.append(s)
            if not diffs:
                print(f"  {lab:22s} (no start-matched {bname} runs to pair)")
                continue
            d = np.asarray(diffs)
            pt = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))
            sd = np.asarray(seed_d)
            st = (sd.mean() / (sd.std(ddof=1) / math.sqrt(len(sd)))
                  if len(sd) > 1 else float("nan"))
            flag = f"  [!{unmatched} start-mismatch skipped]" if unmatched else ""
            print(f"  {lab:22s} diff {d.mean():+.2f}  per-roll t={pt:.2f}  "
                  f"seed t={st:.2f}  ({len(used)} seeds, n={len(d)}){flag}")


if __name__ == "__main__":
    main()
