"""scripts/analyze_kitchen_cg.py

Pooled + paired analysis of the kitchen CG dose-response on the FLAT (MCSS)
DF arms — the analysis behind the "guidance lifts the flat baseline" claim
(results_chapter §6). Torch-free; reads results/kitchen*.json.

Grouping is by the JSON's internal cg_w field, never by filename (a filename
collision already burned one run — the payload is authoritative; see runbook).

Pairing rationale: runs with the same --seed and n_envs share the env reset
sequence, and MCSS executes first in every run with an identical RNG stream —
so env i of a guided run and env i of an unguided run are matched instances,
and per-env differences are a legitimate paired vector. Pairing uses only
n_envs=25 x n_episodes=1 runs (the arm protocol) with matching seeds.

Run (any box):  python scripts/analyze_kitchen_cg.py
"""
import glob
import json
import math
import sys
from collections import defaultdict

sys.path.insert(0, ".")


def load_runs():
    runs = []
    for f in sorted(glob.glob("results/kitchen*.json")):
        try:
            d = json.load(open(f))
        except Exception as exc:
            print(f"  (skipped {f}: {exc!r})")
            continue
        if d.get("backbone") != "df":
            continue                       # the CG claim is about the DF flat arm
        # EXCLUDE non-critic-valued MCSS arms: a grounded run (value_mode=grounded
        # / grounded_mcss) has cg_w=0 but its MCSS arm is grounded-reranked (~55),
        # NOT the plain DF-MCSS baseline (~60). Left in, it collides with the plain
        # unguided run on the (cg_w, seed) key and contaminates every guidance diff.
        if d.get("value_mode") not in (None, "critic") or d.get("grounded_mcss"):
            continue
        r = d.get("results", {}).get("mcss", {})
        v = r.get("dv_norm")
        if not v:
            continue
        runs.append(dict(file=f, seed=d.get("seed"),
                         cg_w=float(d.get("cg_w") or 0.0),
                         k=d.get("k_mcss"), n_envs=d.get("n_envs"),
                         n_eps=d.get("n_episodes"),
                         v=[float(x) for x in v]))
    return runs


def mean_sem(v):
    n = len(v)
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
    return m, sd / math.sqrt(n), n


def main():
    runs = load_runs()
    if not runs:
        sys.exit("no DF-backbone kitchen mcss arms found under results/")

    print("runs found (DF backbone, mcss arm):")
    for r in runs:
        print(f"  {r['file']:44s} seed={r['seed']} cg_w={r['cg_w']:g} "
              f"k={r['k']} n={len(r['v'])}")

    # ── pooled per guidance weight (k=150 arms only — the protocol config) ──
    by_w = defaultdict(list)
    for r in runs:
        if r["k"] == 150:
            by_w[r["cg_w"]] += r["v"]
    print("\npooled flat DF-MCSS by cg_w (k=150 arms):")
    for w in sorted(by_w):
        m, se, n = mean_sem(by_w[w])
        print(f"  cg_w={w:<4g} mean={m:6.2f} +/- {se:4.2f}  (n={n})")

    # ── paired guided-vs-unguided per seed (25x1 arms, matched env indices) ──
    arm = {}
    for r in runs:
        if r["k"] == 150 and r["n_envs"] == 25 and r["n_eps"] == 1:
            key = (r["cg_w"], r["seed"])
            if key in arm:
                print(f"  (note: duplicate arm {key}; keeping {arm[key]['file']}, "
                      f"ignoring {r['file']})")
            else:
                arm[key] = r
    ws = sorted({w for w, _ in arm if w != 0.0})
    for w in ws:
        diffs, seeds = [], []
        for (w2, s), r in sorted(arm.items()):
            if w2 != w or (0.0, s) not in arm:
                continue
            base = arm[(0.0, s)]
            diffs += [a - b for a, b in zip(r["v"], base["v"])]
            seeds.append(s)
        if not diffs:
            print(f"\ncg_w={w:g}: no seed with both guided and unguided 25x1 "
                  f"arms — pooled means above are the comparison")
            continue
        m, se, n = mean_sem(diffs)
        t = m / se if se > 0 else float("inf")
        wl = (sum(1 for x in diffs if x > 0), sum(1 for x in diffs if x < 0),
              sum(1 for x in diffs if x == 0))
        print(f"\npaired guided(w={w:g}) - unguided, matched envs, seeds {seeds}:")
        print(f"  diff={m:+.2f} +/- {se:.2f}  paired t={t:.2f}  n={n}  "
              f"win/loss/tie={wl[0]}/{wl[1]}/{wl[2]}")
        print(f"  (25-pt units: +1 subtask in {wl[0]}/{n} matched envs)")


if __name__ == "__main__":
    main()
