"""scripts/analyze_failures.py

Attribution + characterisation of the MCSS failures dumped by
scripts/run_instrumentation.py.

The headline ATTRIBUTION comes from the ORACLE counterfactual (Tier 2), not the
trajectory-shape classifier. Reason (learned the hard way on the first GPU run):
shape conflates symptom with cause — an end-of-episode topple or a stalled
progress curve is usually the *consequence* of an earlier ranking error, not an
independent execution/horizon failure, and no threshold can separate the two from
shape alone. The oracle can: it changes ONLY the ranking over the same candidates,
so "oracle solves it" == "a better critic solves it" by construction.

So this script:
  * HEADLINE = oracle-fixed / oracle-immune split (the trustworthy critic-fixable %).
  * Tier-1 modes are reported DESCRIPTIVELY (how failures look), not as the
    fixable number.
  * It then characterises the oracle-IMMUNE set — the real ceiling blockers — where
    shape IS informative and uncomfounded, with a per-episode evidence dump.
  * The mode×oracle cross-tab + forward/reverse checks remain, as validation/colour.

    python scripts/analyze_failures.py --in-dir results/instr
    python scripts/analyze_failures.py --in-dir results/instr --out results/instr/summary.json

⚠ All oracle numbers are DIAGNOSTIC-ONLY (Rule-1) — an upper bound, never reportable.
"""
import argparse
import glob
import json
import math
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, ".")

from mcts.failure_modes import (ClassifierConfig, classify_failure, tally,
                                CRITIC_FIXABLE, CRITIC_IMMUNE)
from mcts.instrument import record_from_npz


def _load_runs(in_dir: str, tag: str) -> List[Tuple[dict, str]]:
    """(index dict, npz path) for every {tag}_s*_index.json in in_dir, seed-sorted."""
    out = []
    for ipath in sorted(glob.glob(os.path.join(in_dir, f"{tag}_s*_index.json"))):
        with open(ipath) as f:
            idx = json.load(f)
        out.append((idx, os.path.join(in_dir, idx["npz"])))
    out.sort(key=lambda r: r[0]["seed"])
    return out


def _far_threshold(runs, q: float) -> float:
    """q-th percentile of start geodesic distance across ALL scenarios (far = top tail)."""
    ds = [s["start_geo_cells"] for idx, _ in runs for s in idx["scenarios"]
          if s.get("start_geo_cells") is not None]
    return float(np.percentile(ds, q)) if ds else math.inf


def tier1(runs, cfg: ClassifierConfig, far_q: float):
    """Classify every failure; return (per-scenario records, modes list, totals)."""
    far_thr = _far_threshold(runs, far_q)
    n_total = sum(len(idx["scenarios"]) for idx, _ in runs)
    rows = []           # (seed, env_idx, mode, evidence, scen)
    for idx, npz_path in runs:
        npz = np.load(npz_path, allow_pickle=False)
        for scen in idx["scenarios"]:
            if scen["success"]:
                continue
            i = scen["env_idx"]
            far = (scen.get("start_geo_cells") is not None
                   and scen["start_geo_cells"] >= far_thr)
            rec = record_from_npz(npz, i, reach_step=scen.get("reach_step"),
                                  is_far=far, goal=scen.get("goal"))
            if rec is None:
                rows.append((idx["seed"], i, "UNCLASSIFIED", {}, scen))
                continue
            mode, ev = classify_failure(rec, cfg)
            rows.append((idx["seed"], i, mode, ev, scen))
    modes = [r[2] for r in rows]
    return rows, modes, n_total, far_thr


def _pair_oracle(rows, oracle_runs):
    """Annotate each failure row with the oracle outcome: True=fixed, False=immune,
    None=unpaired. Returns (annotated_rows, have_oracle)."""
    osucc: Dict[Tuple[int, int], bool] = {}
    for idx, _ in oracle_runs:
        for s in idx["scenarios"]:
            osucc[(idx["seed"], s["env_idx"])] = s["success"]
    annotated = [(seed, i, mode, ev, scen, osucc.get((seed, i)))
                 for seed, i, mode, ev, scen in rows]
    have_oracle = any(o is not None for *_, o in annotated)
    return annotated, have_oracle


def _feat(ev, name, default=float("nan")):
    f = ev.get("features")
    return getattr(f, name, default) if f is not None else default


def paired_oracle(runs, oracle_runs):
    """PAIRED oracle effect over ALL scenarios (not just critic failures): the ceiling
    must net the breaks (critic-success -> oracle-fail), or 'fixes-only' inflates it
    exactly the way unpaired noise did at n=500. Uses only the two index files."""
    csucc = {(idx["seed"], s["env_idx"]): bool(s["success"])
             for idx, _ in runs for s in idx["scenarios"]}
    osucc = {(idx["seed"], s["env_idx"]): bool(s["success"])
             for idx, _ in oracle_runs for s in idx["scenarios"]}
    keys = sorted(set(csucc) & set(osucc))
    if not keys:
        return None
    n = len(keys)
    fixes = sum(1 for k in keys if not csucc[k] and osucc[k])
    breaks = sum(1 for k in keys if csucc[k] and not osucc[k])
    return dict(n=n, fixes=fixes, breaks=breaks, net=fixes - breaks,
                critic_reach=100.0 * sum(csucc[k] for k in keys) / n,
                oracle_reach=100.0 * sum(osucc[k] for k in keys) / n)


def print_headline(annotated, n_total, base_reach, have_oracle, env, paired):
    """The trustworthy attribution: the PAIRED oracle counterfactual (fixes AND breaks)."""
    n_fail = len(annotated)
    print("\n" + "=" * 72)
    print(f"HEADLINE — oracle counterfactual, PAIRED over all scenarios  ({env})")
    if not have_oracle or paired is None:
        print(f"  MCSS baseline reach : {base_reach:.1f}%   ({n_fail}/{n_total} fail)")
        print("  NO ORACLE RUN found (instr_mcss_oracle_s*_index.json). The fixable %")
        print("  is the oracle's to give — run scripts/run_instrumentation.py "
              "--value-source oracle.")
        return None
    p = paired
    print(f"  critic reach : {p['critic_reach']:.1f}%        (paired n={p['n']})")
    print(f"  oracle reach : {p['oracle_reach']:.1f}%        (the oracle run's ACTUAL reach)")
    print(f"  fixes        : {p['fixes']:>3}   (critic fail -> oracle solve)")
    print(f"  breaks       : {p['breaks']:>3}   (critic solve -> oracle FAIL "
          f"-- the number the fixes-only ceiling ignored)")
    netpp = 100.0 * p['net'] / p['n']
    print(f"  NET          : {p['net']:+d} rollouts = {netpp:+.1f} pp  "
          f"=> honest ceiling {p['oracle_reach']:.1f}%")
    if p['fixes'] > 0:
        ratio = p['breaks'] / p['fixes']
        if ratio >= 0.6:
            print(f"  ** breaks/fixes = {ratio:.2f}: the apparent +{p['fixes']} 'fix' is "
                  f"largely re-sampling NOISE, not a genuine ranking gain.")
        else:
            print(f"  breaks/fixes = {ratio:.2f}: net gain survives the breaks "
                  f"-> a genuine (privileged) ranking advantage.")
    immune = sum(1 for *_, o in annotated if o is False)
    print(f"  oracle-IMMUNE failures (ceiling blockers, characterised below): {immune}/{n_fail}")
    print("  Rule-1: oracle uses true geodesics -> upper bound, NOT reportable.")
    return dict(critic_reach=p['critic_reach'], oracle_reach=p['oracle_reach'],
                fixes=p['fixes'], breaks=p['breaks'], net=p['net'], immune=immune)


def print_mode_mix(modes, n_total):
    """Tier-1 descriptive mode mix — how the failures LOOK (no fixable claim)."""
    print("\n" + "-" * 72)
    print("TIER 1 — failure-mode mix (DESCRIPTIVE)")
    print(f"  {'mode':<20} {'count':>6} {'pct(fail)':>10} {'pct(all)':>9}")
    for mode, c, pct in tally(modes):
        print(f"  {mode:<20} {c:>6} {pct:>9.1f}% {100.0*c/n_total:>8.1f}%")


def print_crosstab(annotated):
    """mode × oracle outcome — validation/colour (and the calibration warnings)."""
    by_mode: Dict[str, List[int]] = {}
    for seed, i, mode, ev, scen, o in annotated:
        cell = by_mode.setdefault(mode, [0, 0])
        if o is True:
            cell[0] += 1
        elif o is False:
            cell[1] += 1
    print("\n  CROSS-CHECK  (Tier-1 mode x oracle outcome):")
    print(f"  {'mode':<20} {'oracle-fixed':>12} {'oracle-fails':>12}")
    print("  " + "-" * 46)
    for mode in sorted(by_mode, key=lambda m: -(by_mode[m][0] + by_mode[m][1])):
        ff, im = by_mode[mode]
        print(f"  {mode:<20} {ff:>12} {im:>12}")
    # forward: do the modes we CALL fixable actually get oracle-fixed?
    cf, ct = 0, 0
    for mode, (ff, im) in by_mode.items():
        if mode in CRITIC_FIXABLE:
            cf += ff
            ct += ff + im
    if ct:
        rate = 100.0 * cf / ct
        print(f"\n  forward: {rate:.0f}% of CRITIC_FIXABLE-mode failures are oracle-fixed "
              f"({'consistent' if rate >= 60 else 'low — shape under-detects ranking'}).")
    # reverse: any IMMUNE mode the oracle fixes a lot == shape mislabeled a ranking failure
    rev = [f"{m} {by_mode[m][0]}/{sum(by_mode[m])} ({100*by_mode[m][0]/sum(by_mode[m]):.0f}%)"
           for m in sorted(by_mode)
           if m in CRITIC_IMMUNE and sum(by_mode[m]) >= 4
           and by_mode[m][0] / sum(by_mode[m]) > 0.5]
    if rev:
        print("  reverse: shape mislabels ranking failures as immune (oracle fixes them): "
              + "; ".join(rev))
        print("           -> trust the oracle headline, not these mode labels.")
    else:
        print("  reverse: no immune mode is over-fixed by the oracle (modes track the oracle).")
    return by_mode, rev


def print_immune_evidence(annotated):
    """Characterise the oracle-IMMUNE failures — the real ceiling blockers, where
    trajectory shape IS informative (the oracle cannot reach them by re-ranking, so
    they are genuinely proposal/execution/horizon limited)."""
    immune = [(s, i, m, ev, scen) for s, i, m, ev, scen, o in annotated if o is False]
    print("\n" + "-" * 72)
    print(f"ORACLE-IMMUNE FAILURES — the ceiling blockers ({len(immune)})")
    if not immune:
        print("  (none — every failure is oracle-fixable by re-ranking)")
        return []
    for mode, c, pct in tally([m for _, _, m, _, _ in immune]):
        print(f"    {mode:<20} {c:>3}  ({pct:.0f}% of immune)")
    print(f"  {'scn':<8} {'mode':<16} {'startGeo':>8} {'min':>5} {'end':>5} "
          f"{'bkslide':>7} {'offg':>5} {'collapse':>8} {'wMin':>6}  why")
    rows_out = []
    for s, i, mode, ev, scen in sorted(immune, key=lambda r: (r[0], r[1])):
        wmin = ev.get("min_world_dist")
        coll = ev.get("collapse_step")
        line = (f"  s{s}e{i:<6} {mode:<16} "
                f"{_fmt(scen.get('start_geo_cells')):>8} {_fmt(_feat(ev,'min_dist')):>5} "
                f"{_fmt(_feat(ev,'end_dist')):>5} {_fmt(_feat(ev,'backslide')):>7} "
                f"{_feat(ev,'off_graph_frac',0.0):>5.2f} {str(coll):>8} "
                f"{_fmt(wmin):>6}  {ev.get('why','')}")
        print(line)
        rows_out.append(dict(seed=s, env_idx=i, mode=mode,
                             start_geo_cells=scen.get("start_geo_cells"),
                             min_dist=_feat(ev, "min_dist"), end_dist=_feat(ev, "end_dist"),
                             backslide=_feat(ev, "backslide"),
                             off_graph_frac=_feat(ev, "off_graph_frac", 0.0),
                             collapse_step=coll, min_world_dist=wmin, why=ev.get("why")))
    return rows_out


def _fmt(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    return f"{x:.0f}" if isinstance(x, (int, float)) else str(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=str, default="results/instr")
    p.add_argument("--critic-tag", type=str, default="instr_mcss_critic")
    p.add_argument("--oracle-tag", type=str, default="instr_mcss_oracle")
    p.add_argument("--far-quantile", type=float, default=80.0,
                   help="start-distance percentile above which a failure is 'far'")
    p.add_argument("--out", type=str, default=None, help="optional JSON summary path")
    args = p.parse_args()

    runs = _load_runs(args.in_dir, args.critic_tag)
    if not runs:
        sys.exit(f"no critic runs found: {args.in_dir}/{args.critic_tag}_s*_index.json")
    env = runs[0][0]["env"]
    base_reach = float(np.mean([idx["reach_pct"] for idx, _ in runs]))
    cfg = ClassifierConfig()

    rows, modes, n_total, far_thr = tier1(runs, cfg, args.far_quantile)
    oracle_runs = _load_runs(args.in_dir, args.oracle_tag)
    annotated, have_oracle = _pair_oracle(rows, oracle_runs)
    paired = paired_oracle(runs, oracle_runs) if oracle_runs else None

    headline = print_headline(annotated, n_total, base_reach, have_oracle, env, paired)
    print_mode_mix(modes, n_total)
    by_mode = rev = None
    immune_rows = []
    if have_oracle:
        by_mode, rev = print_crosstab(annotated)
        immune_rows = print_immune_evidence(annotated)
    print("=" * 72)

    if args.out:
        summary = dict(
            env=env, n_total=n_total, n_failed=len(modes), base_reach=base_reach,
            far_threshold=far_thr,
            headline=headline,                       # oracle attribution (the number to use)
            tier1_tally=tally(modes),                # descriptive mode mix
            crosstab={m: dict(oracle_fixed=v[0], oracle_fails=v[1])
                      for m, v in (by_mode or {}).items()},
            reverse_flags=rev or [],
            oracle_immune=immune_rows,               # the ceiling blockers, per episode
            per_scenario=[dict(seed=s, env_idx=i, mode=m,
                               oracle_fixed=o,
                               start_geo_cells=scen.get("start_geo_cells"),
                               reach_step=scen.get("reach_step"))
                          for s, i, m, ev, scen, o in annotated])
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
