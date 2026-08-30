"""scripts/collate_mctd.py

Paired MCTD-vs-MCSS aggregation for the D4RL MCTD study (scripts/run_mctd.py +
scripts/run_mcts_compare.py --method mcss). MCTD and MCSS run at the SAME
--env/--seed/--n-envs see identical (seed, index) starts+goals, so per-rollout
values pair by index within a seed, and per-seed means pair across seeds.

Reports, on BOTH metrics (reach% and the DV camping score):
  * per-seed means for each method,
  * the SEED-LEVEL paired t (mean over seeds of the per-seed difference / its SE)
    — the primary statistic (the runbook's standard),
  * the per-rollout paired t pooled over all seed x index (secondary).

Torch-free (json + math). Verifies the goal vectors match before pairing, so a
seed mismatch is caught loudly rather than producing a bogus paired test.

Usage:
    python scripts/collate_mctd.py \
        --mctd "results/mctd_maze2d_large_rp50_s*.json" \
        --mcss "results/mcss_maze2d_large_s*.json"
"""
import argparse
import glob
import json
import math
import sys
from typing import Dict, List, Optional, Tuple


def _load(path: str) -> Tuple[int, str, dict]:
    d = json.load(open(path))
    seed = d.get("seed")
    res = d.get("results", {})
    method = next(iter(res)) if res else None       # "mctd" or "mcss"
    return seed, method, res.get(method, {})


def _by_seed(paths: List[str]) -> Dict[int, dict]:
    out = {}
    for p in sorted(paths):
        seed, method, r = _load(p)
        if not r:
            print(f"  (skipped {p}: no results payload)")
            continue
        if seed in out:
            print(f"  (note: duplicate seed {seed} for {method}; keeping first)")
            continue
        rec = dict(r)                       # r already carries its own "method" key
        rec["file"] = p
        rec.setdefault("method", method)
        out[seed] = rec
    return out


def mean_sem(v: List[float]) -> Tuple[float, float, int]:
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
    return m, sd / math.sqrt(n) if n > 1 else 0.0, n


def paired_t(diffs: List[float]) -> Tuple[float, float, int]:
    m, se, n = mean_sem(diffs)
    t = m / se if se > 0 else float("inf") if m != 0 else 0.0
    return m, t, n


def _goals_match(ga, gb) -> bool:
    if ga is None or gb is None or len(ga) != len(gb):
        return False
    for a, b in zip(ga, gb):
        if a is None or b is None:
            continue
        if abs(a[0] - b[0]) > 1e-6 or abs(a[1] - b[1]) > 1e-6:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mctd", required=True, help="glob for MCTD result JSONs")
    ap.add_argument("--mcss", required=True, help="glob for MCSS result JSONs")
    ap.add_argument("--metric", default="dv_norm", choices=["dv_norm", "success"],
                    help="per-rollout vector to pair (dv_norm = camping score)")
    args = ap.parse_args()

    mctd = _by_seed(glob.glob(args.mctd))
    mcss = _by_seed(glob.glob(args.mcss))
    seeds = sorted(set(mctd) & set(mcss))
    if not seeds:
        sys.exit(f"no shared seeds between MCTD ({sorted(mctd)}) and "
                 f"MCSS ({sorted(mcss)})")
    print(f"paired seeds: {seeds}")

    # ── seed-level means (primary) + pooled per-rollout diffs (secondary) ──
    seed_diff_camp, seed_diff_reach = [], []
    pooled_diff = []
    print(f"\n{'seed':>4} {'MCTD camp':>10} {'MCSS camp':>10} {'Δcamp':>7} "
          f"{'MCTD reach':>10} {'MCSS reach':>10} {'Δreach':>7} {'n':>4}")
    for s in seeds:
        a, b = mctd[s], mcss[s]
        if not _goals_match(a.get("goals"), b.get("goals")):
            print(f"  WARNING seed {s}: goal vectors differ — MCTD and MCSS did "
                  f"NOT see the same scenarios; pairing invalid for this seed")
        va, vb = a[args.metric], b[args.metric]
        if len(va) != len(vb):
            print(f"  WARNING seed {s}: length mismatch {len(va)} vs {len(vb)}; "
                  f"truncating to min")
        n = min(len(va), len(vb))
        pooled_diff += [va[i] - vb[i] for i in range(n)]
        ca, _, _ = mean_sem(a["dv_norm"])
        cb, _, _ = mean_sem(b["dv_norm"])
        ra, rb = a["reach_pct"], b["reach_pct"]
        seed_diff_camp.append(ca - cb)
        seed_diff_reach.append(ra - rb)
        print(f"{s:>4} {ca:>10.1f} {cb:>10.1f} {ca - cb:>+7.1f} "
              f"{ra:>10.1f} {rb:>10.1f} {ra - rb:>+7.1f} {n:>4}")

    print("\n" + "=" * 60)
    mc, tc, nc = paired_t(seed_diff_camp)
    mr, tr, nr = paired_t(seed_diff_reach)
    print(f"SEED-LEVEL paired (n={nc} seeds):")
    print(f"  camping  MCTD-MCSS = {mc:+.2f}  t={tc:+.2f}")
    print(f"  reach%   MCTD-MCSS = {mr:+.2f}  t={tr:+.2f}")
    mp, tp, npd = paired_t(pooled_diff)
    print(f"per-rollout pooled ({args.metric}, n={npd}): "
          f"diff={mp:+.2f}  t={tp:+.2f}")
    verdict = ("MCTD BEATS MCSS" if tc > 2 else "MCTD LOSES to MCSS"
               if tc < -2 else "NULL (|t|<2): MCTD does not beat MCSS")
    print(f"\n  verdict (seed-level camping): {verdict}")
    print("=" * 60)


if __name__ == "__main__":
    main()
