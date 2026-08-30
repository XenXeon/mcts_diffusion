"""scripts/collate_mcts.py

Collate run_mcts_compare JSON outputs into one candidates-per-step vs reach% table,
and run exact McNemar paired tests wherever paired per-rollout vectors are available.

Compute accounting: MCTS at budget B with k_mcts candidates evaluates k_mcts*(B+1)
plans per step, while MCSS evaluates k_mcss plans per step. Sorting both by
"candidates/step" lines up the matched-compute comparison (is an MCTS win just more
plans, or the look-ahead structure?).

Pairing: run_episodes stores a per-rollout `success` vector (episode-major; with the
same seed and n_envs, index i is the same start/goal scenario in every run — see
mcts/mcts_loop.py). Two cells with equal (env, seed) and equal-length vectors are
therefore PAIRED, and the right test is McNemar on the discordant rollouts:
    b = scenarios MCSS missed but MCTS reached   ("fixes")
    c = scenarios MCSS reached but MCTS missed   ("breaks")
Exact two-sided binomial p on (b, c); concordant rollouts carry no information.
Old JSONs without `success` still collate — they just can't be paired.

Error bars: reach% gets the binomial SEM sqrt(p(1-p)/n)*100, computed from the
success vector (or the stored reach_err). The old norm_err is the SEM of the D4RL
normalized score — on antmaze it coincides with the reach SEM, on maze2d it is a
metric artifact; it is no longer printed as the reach error.

Usage:
    python scripts/collate_mcts.py results/mcss_antmaze_*.json results/mcts_antmaze_*.json
    python scripts/collate_mcts.py            # defaults to results/*antmaze*.json + *maze2d*.json
"""
import glob
import json
import math
import sys


def candidates_per_step(d: dict, method: str) -> int:
    if method == "mcts":
        k = int(d.get("k_mcts", 0))
        k_root = int(d.get("k_root", 0) or k)      # wide-root runs record k_root
        return k_root + int(d.get("budget", 0)) * k
    return int(d.get("k_mcss", 0))


# value/gate config -> short label suffix (B2; §5 item 6). Absent keys (every
# naive 2×2 cell, and all pre-hardening cells) -> "" so labels are unchanged.
# "oracle" -> "orc": the §7.5 privileged-value tree (DIAGNOSTIC-ONLY, Rule-1) — its
# cells carry DIAGNOSTIC_ONLY in the JSON; never report the orc number.
_VALUE_TAG = {None: "", "": "", "v_s": "", "v_sg": "sg", "v_sg_pess": "sgP",
              "oracle": "orc", "oracle_fs": "fs", "oracle_fsf": "fsf", "oracle_stb": "stb",
              "oracle_gnt": "gnt", "oracle_smt": "smt", "grounded": ""}


def config_suffix(d: dict) -> str:
    vt = _VALUE_TAG.get(d.get("value_mode"), str(d.get("value_mode")))
    # first-step oracle: distinguish the keep-band sweep (fs0/fs2/fs4/fs8) so the bands
    # do not collapse into one label and merge/skip in the McNemar pairing.
    if d.get("value_mode") == "oracle_fs" and d.get("keep_band") is not None:
        vt = f"fs{int(d['keep_band'])}"
    elif d.get("value_mode") == "oracle_fsf":
        mp = d.get("max_progress_cells", d.get("max_step_cells", 0))   # renamed key (back-compat)
        vt = f"fsf{int(d.get('keep_band', 0))}m{int(mp)}"
    elif d.get("value_mode") == "oracle_stb":
        sb = str(d.get("stability_by", "?"))[0].upper()                # U/D/A
        vt = f"stb{sb}{int(d.get('keep_band', 0))}"
    elif d.get("value_mode") == "oracle_gnt":
        vt = f"gnt{int(float(d.get('lunge_frac', 0)) * 100)}"
    elif d.get("value_mode") == "oracle_smt":
        vt = f"smt{int(float(d.get('turn_cap', 0)))}"
    # Grounded subtask checker (mcts/grounded.py) — a NON-learned evaluator, exempt
    # from the 3-of-4 label cap every learned value inherits from kitchen-mixed's
    # demonstrations. A grounded-valued arm (value_mode="grounded") OR a grounded
    # MCSS rerank (grounded_mcss=True, independent of value_mode) is a DIFFERENT
    # evaluator from every critic-valued arm and must never pool with one under
    # the same base label.
    if d.get("value_mode") == "grounded" or d.get("grounded_mcss"):
        gb_raw = d.get("grounded_blend")   # None on old JSONs -> the run default
        gb = 0.25 if gb_raw is None else float(gb_raw)
        vt += "Gnd" + ("" if gb == 0.25 else f"{gb:g}")
    # critic-tree variants: a widened root (k_root != k_mcts) and/or top-m backup
    # must NOT collapse into the plain b{budget}-critic label, or the table pools
    # different algorithms at the same seed (the naive-vs-composed trap).
    kr, km = d.get("k_root"), d.get("k_mcts")
    if kr and km and int(kr) != int(km):
        vt += f"r{int(kr)}"
    if int(d.get("top_m", 1) or 1) > 1:
        vt += f"m{int(d['top_m'])}"
    # non-default critic checkpoint (e.g. the stitch-aware fine-tune) and the
    # junction filter are separate ARMS — keep their labels distinct too
    cs = str(d.get("critic_step") or "")
    if cs and cs not in ("1000000", "200000"):
        vt += f"C{cs[:6]}"
    # non-default V(s) checkpoint (e.g. the distilled plan-value 'planv') is a
    # different value function -> must not collapse into the plain v_s label
    vs = str(d.get("value_step") or "")
    if d.get("value_mode") == "v_s" and vs not in ("", "latest", "best"):
        vt += f"V{vs[:5]}"
    if d.get("junction_filter"):
        vt += "J"
    # prefix-inpainted expansion (DF-inspired) is a different search MECHANISM,
    # not just a knob — old JSONs lack the key and keep their existing labels
    if d.get("expand_mode") == "inpaint":
        vt += "inp"
    # Diffusion Forcing backbone: a different PLANNER — never pool with DV arms.
    # Applies to both mcss and mcts columns of a --df-ckpt run.
    if d.get("backbone") == "df":
        tag = str(d.get("df_ckpt") or "")
        vt += "DF" + ("" if tag in ("", "final") else tag[:5])
        # Classifier guidance (--cg-w) is a different SAMPLING ALGORITHM on the
        # same DF backbone — different guidance = different arm, so a cg_w!=0
        # run must never pool with a plain (unguided) DF run at the same label.
        cg_w = d.get("cg_w") or 0
        if cg_w:
            vt += f"cg{float(cg_w):g}"
    gt = "g" if d.get("gate") == "hard" else ""
    return f"-{vt}{gt}" if (vt or gt) else ""


def binom_err_pct(p_frac: float, n: int) -> float:
    """Binomial SEM of a success rate, in percentage points."""
    if n <= 0:
        return float("nan")
    return math.sqrt(max(p_frac * (1.0 - p_frac), 0.0) / n) * 100.0


def mcnemar_exact(a, b):
    """Exact two-sided McNemar test for paired 0/1 outcomes.

    a, b: equal-length sequences of 0/1 in the same (seed, rollout-index) order.
    Returns (n01, n10, p): n01 = a-miss/b-reach ("fixes" if b is the new method),
    n10 = a-reach/b-miss ("breaks"), and the exact two-sided binomial p-value on
    the discordant count split (p=1.0 when there are no discordant pairs).
    """
    if len(a) != len(b):
        raise ValueError(f"paired vectors differ in length: {len(a)} vs {len(b)}")
    n01 = sum(1 for x, y in zip(a, b) if not x and y)
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * 0.5 ** n
    return n01, n10, min(1.0, 2.0 * tail)


def load_rows(paths):
    rows = []
    for p in paths:
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception as e:
            print(f"skip {p}: {e}")
            continue
        for method, r in d.get("results", {}).items():
            # Label must distinguish child_index AND value/gate variants — else
            # cells at the same env+seed collide in the table and become
            # indistinguishable in the McNemar pairing rows. The naive 2×2 cells
            # (V(s) / DV critic, no gate) carry no value/gate keys ⇒ no suffix ⇒
            # stay b16/k272; hardened cells (B2) get a suffix the moment
            # run_mcts_compare writes value_mode/gate.
            cidx = int(d.get("child_index", 1) or 1)
            suffix = config_suffix(d)
            if method == "mcts":
                label = (f"b{d.get('budget')}"
                         + (f"L{cidx}" if cidx != 1 else "") + suffix)
            else:
                label = f"k{d.get('k_mcss')}" + suffix
            success = r.get("success")          # per-rollout 0/1, None on old JSONs
            n = int(r.get("n_rollouts", 0))
            reach = r.get("reach_pct", float("nan"))
            if success is not None and len(success) > 0:
                err = binom_err_pct(sum(success) / len(success), len(success))
            elif "reach_err" in r:
                err = float(r["reach_err"])
            else:                               # legacy fallback: norm_err ≈ reach SEM on antmaze
                err = float(r.get("norm_err", float("nan")))
            rows.append(dict(env=d.get("env", "?"), seed=d.get("seed"),
                             method=method, label=label,
                             value_mode=d.get("value_mode"), gate=d.get("gate"),
                             cands=candidates_per_step(d, method),
                             reach=reach, err=err, n=n,
                             # DV-exact score: reach% on antmaze, camping return (>100) on
                             # maze2d. dv_vec is the per-rollout vector to PAIR on for maze2d,
                             # where the binary reach above is saturated (see print_camping).
                             dv_score=float(r.get("dv_norm_mean", reach)),
                             dv_err=float(r.get("dv_norm_err", err)),
                             dv_vec=r.get("dv_norm"),
                             wall=r.get("wall_s", 0.0),
                             depth=r.get("tree_depth_mean"),
                             # Rule-1: privileged-oracle cells carry DIAGNOSTIC_ONLY; mark
                             # them in the table so the number can't be mis-cited as achievable.
                             diag=bool(d.get("DIAGNOSTIC_ONLY", False)),
                             success=success,
                             goals=r.get("goals"), starts=r.get("starts"), file=p))
    return rows


# Canonical baseline→treatment ordering so the McNemar direction is FIXED, not
# data-dependent (F1): less-treated arm is the baseline. mcss < mcts; within a
# method v_s < v_sg < v_sg_pess < oracle, no-gate < hard-gate; then compute then label.
# "oracle" ranks highest so the privileged-value tree is always the treatment, and
# every learned arm reads as its baseline (the §7.5 attribution ladder).
_VALUE_RANK = {None: 0, "": 0, "v_s": 0, "v_sg": 1, "v_sg_pess": 2,
               "oracle": 3, "oracle_fs": 3, "oracle_fsf": 3, "oracle_stb": 3,
               "oracle_gnt": 3, "oracle_smt": 3}


def _precedence(x):
    return (0 if x["method"] == "mcss" else 1,
            _VALUE_RANK.get(x.get("value_mode"), 0),
            1 if x.get("gate") == "hard" else 0,
            x["cands"], str(x["label"]))


def pairing_check(a: dict, b: dict, tol: float = 1e-3):
    """Verify two cells really are paired, using the stored per-rollout vectors.

    The scenario-defining variable is the GOAL (antmaze jitters it per episode
    around the eval corner): per-index goals must match within `tol`, otherwise
    index i is not the same scenario in both runs and McNemar is invalid.
    Start positions are reset noise around a fixed cell; their max per-index
    divergence is reported informationally (large values indicate sub-env
    seeding is not deterministic — see mcts/mcts_loop.py run_episodes).

    Returns (ok, note): ok=False means goals demonstrably differ.
    """
    ga, gb = a.get("goals") or [], b.get("goals") or []
    pairs = [(x, y) for x, y in zip(ga, gb) if x is not None and y is not None]
    if not (ga and gb and len(ga) == len(gb) and pairs):
        return True, "goals not recorded (pairing unverified)"
    bad = sum(1 for x, y in pairs
              if max(abs(x[0] - y[0]), abs(x[1] - y[1])) > tol)
    if bad:
        return False, f"GOALS DIFFER on {bad}/{len(pairs)} indices"
    note = "goals match"
    sa, sb = a.get("starts") or [], b.get("starts") or []
    if sa and sb and len(sa) == len(sb):
        d = max(max(abs(x[0] - y[0]), abs(x[1] - y[1])) for x, y in zip(sa, sb))
        note += f", start jitter {d:.3f}"
        if d > 0.5:
            note += " (>0.5: seeding suspect)"
    return True, note


def is_maze2d(env: str) -> bool:
    return str(env).startswith("maze2d")


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def paired_camping(a_vec, b_vec):
    """Paired comparison of the DV camping return (maze2d), where binary reach is saturated
    so McNemar is degenerate. Per-scenario differences d_i = b_i - a_i; returns
    (mean_a, mean_b, mean_delta, sem, two-sided normal-approx p). Reports the DV metric the
    way the paper does (normalized score), with a paired significance test over shared
    scenarios instead of a reach indicator."""
    d = [float(b) - float(a) for a, b in zip(a_vec, b_vec)]
    n = len(d)
    if n == 0:
        return (float("nan"),) * 5
    ma, mb = sum(a_vec) / n, sum(b_vec) / n
    md = sum(d) / n
    var = sum((x - md) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    sem = math.sqrt(var / n) if n > 0 else float("nan")
    z = md / sem if sem > 0 else 0.0
    p = 2.0 * (1.0 - _norm_cdf(abs(z)))
    return ma, mb, md, sem, p


def print_table(rows):
    # DV-score column: the base-pipeline metric — reach% on antmaze, camping return (>100)
    # on maze2d. On maze2d this, NOT reach%, is the DV number to report and pair on.
    print(f"{'env':>26} {'method':>6} {'cfg':>8} {'seed':>4} {'cand/step':>9} "
          f"{'reach%':>7} {'DV-score':>9} {'err%':>6} {'n':>4} {'depth':>6} {'wall(s)':>8} {'paired?':>7}")
    print("-" * 116)
    any_diag = False
    for x in rows:
        depth = f"{x['depth']:.1f}" if x.get("depth") is not None else "-"
        # mark privileged-oracle rows (Rule-1) with '*' in the DISPLAY only — the stored
        # x['label'] stays unmarked so McNemar pairing keys are unaffected.
        lbl = x["label"] + ("*" if x.get("diag") else "")
        any_diag = any_diag or x.get("diag")
        print(f"{x['env'][:26]:>26} {x['method']:>6} {lbl:>8} "
              f"{str(x['seed']):>4} {x['cands']:>9} "
              f"{x['reach']:>6.1f} {x.get('dv_score', float('nan')):>9.1f} {x['err']:>6.1f} "
              f"{x['n']:>4} {depth:>6} {x['wall']:>8.0f} "
              f"{'yes' if x['success'] else 'no':>7}")
    if any(is_maze2d(x["env"]) for x in rows):
        print("  maze2d: report DV-score (camping return, >100) — reach% is saturated there.")
    if any_diag:
        print("  * DIAGNOSTIC-ONLY (Rule-1): privileged true-geodesic value — a CEILING "
              "probe, NOT an achievable/reportable number.")


def print_mcnemar(rows):
    """Per-seed AND pooled paired McNemar over distinct-label cells (F1).

    The baseline `a` is fixed by _precedence (the less-treated arm), NOT by reach,
    so the direction is constant across seeds: `fixes` = a-miss/b-reach, dreach>0
    always means the treatment `b` beat the baseline `a`. The pooled rows
    concatenate each arm's paired success vectors across the shared seeds in that
    canonical direction — the n=150 headline statistic. pairing_check verifies
    per-index goal identity before any pooling.
    """
    paired = [x for x in rows if x["success"]]
    keys = sorted({(x["env"], x["seed"]) for x in paired})
    valid = []   # (env, seed, a, b, note) canonical pairs that passed every check
    for env, seed in keys:
        cells = [x for x in paired if (x["env"], x["seed"]) == (env, seed)]
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                a, b = sorted((cells[i], cells[j]), key=_precedence)
                if a["label"] == b["label"]:
                    continue
                if len(a["success"]) != len(b["success"]):
                    print(f"  [skip] {env} seed={seed} {a['label']} vs {b['label']}: "
                          f"n mismatch ({len(a['success'])} vs {len(b['success'])})")
                    continue
                ok, note = pairing_check(a, b)
                if not ok:
                    print(f"  [INVALID] {env} seed={seed} {a['label']}->{b['label']}: "
                          f"{note} -- not the same scenarios; McNemar not computed.")
                    continue
                valid.append((env, seed, a, b, note))
    if not valid:
        print("\n(no pairable cells: need >=2 distinct-label cells at the same "
              "env+seed, both with per-rollout success vectors)")
        return
    valid_mc = [v for v in valid if not is_maze2d(v[0])]   # antmaze: reach -> McNemar
    valid_cp = [v for v in valid if is_maze2d(v[0])]       # maze2d: camping -> paired diff

    if valid_mc:
        print("\nPaired McNemar -- per seed (baseline -> treatment; dreach>0 => treatment better):")
        print(f"{'env':>22} {'seed':>4} {'pair':>18} {'n':>4} {'fixes':>5} "
              f"{'breaks':>6} {'dreach':>7} {'exact p':>9}  pairing")
        print("-" * 112)
        for env, seed, a, b, note in valid_mc:
            f_, br, p = mcnemar_exact(a["success"], b["success"])
            print(f"{env[:22]:>22} {seed:>4} {a['label']+'->'+b['label']:>18} "
                  f"{len(a['success']):>4} {f_:>5} {br:>6} "
                  f"{b['reach']-a['reach']:>+6.1f}pp {p:>9.4f}  {note}")
        # ── Pooled across seeds, canonical direction -- the headline statistic ───
        groups: dict = {}
        for env, seed, a, b, note in valid_mc:
            groups.setdefault((env, a["label"], b["label"]), []).append((a, b))
        print("\nPaired McNemar -- POOLED across seeds (the headline n>=150 statistic):")
        print(f"{'env':>22} {'pair':>18} {'seeds':>5} {'n':>5} {'reach a->b':>14} "
              f"{'fixes':>5} {'breaks':>6} {'exact p':>9}")
        print("-" * 96)
        for (env, al, bl), plist in sorted(groups.items()):
            A = [v for a, _ in plist for v in a["success"]]
            B = [v for _, b in plist for v in b["success"]]
            f_, br, p = mcnemar_exact(A, B)
            ra, rb = 100 * sum(A) / len(A), 100 * sum(B) / len(B)
            print(f"{env[:22]:>22} {al+'->'+bl:>18} {len(plist):>5} {len(A):>5} "
                  f"{f'{ra:.1f}->{rb:.1f}':>14} {f_:>5} {br:>6} {p:>9.4f}")

    if valid_cp:
        # maze2d: binary reach is saturated -> pair on the DV camping return (the paper's
        # normalized score) with a per-scenario paired-difference test.
        usable = [v for v in valid_cp if v[2].get("dv_vec") and v[3].get("dv_vec")]
        missing = len(valid_cp) - len(usable)
        print("\nPaired DV camping score (maze2d; reach saturated) -- per seed "
              "(baseline -> treatment; d>0 => treatment better):")
        print(f"{'env':>22} {'seed':>4} {'pair':>18} {'n':>4} {'DV a->b':>16} "
              f"{'delta':>8} {'sem':>6} {'p':>8}")
        print("-" * 100)
        for env, seed, a, b, note in usable:
            ma, mb, md, sem, p = paired_camping(a["dv_vec"], b["dv_vec"])
            print(f"{env[:22]:>22} {seed:>4} {a['label']+'->'+b['label']:>18} "
                  f"{len(a['dv_vec']):>4} {f'{ma:.1f}->{mb:.1f}':>16} "
                  f"{md:>+8.1f} {sem:>6.1f} {p:>8.4f}")
        groups2: dict = {}
        for env, seed, a, b, note in usable:
            groups2.setdefault((env, a["label"], b["label"]), []).append((a, b))
        if groups2:
            print("\nPaired DV camping score (maze2d) -- POOLED across seeds:")
            print(f"{'env':>22} {'pair':>18} {'seeds':>5} {'n':>5} {'DV a->b':>16} "
                  f"{'delta':>8} {'sem':>6} {'p':>8}")
            print("-" * 100)
            for (env, al, bl), plist in sorted(groups2.items()):
                A = [v for a, _ in plist for v in a["dv_vec"]]
                B = [v for _, b in plist for v in b["dv_vec"]]
                ma, mb, md, sem, p = paired_camping(A, B)
                print(f"{env[:22]:>22} {al+'->'+bl:>18} {len(plist):>5} {len(A):>5} "
                      f"{f'{ma:.1f}->{mb:.1f}':>16} {md:>+8.1f} {sem:>6.1f} {p:>8.4f}")
        if missing:
            print(f"  ({missing} maze2d pair(s) lack per-rollout dv_norm — rerun "
                  f"run_mcts_compare after this update to enable camping pairing.)")


def main():
    paths = sys.argv[1:]
    if not paths:
        paths = sorted(set(glob.glob("results/*antmaze*.json")
                           + glob.glob("results/*maze2d*.json")))
    rows = load_rows(paths)
    if not rows:
        print("no result JSONs found.")
        return
    rows.sort(key=lambda x: (x["env"], x["method"], x["cands"]))
    print_table(rows)

    # quick matched-compute hint: best MCSS vs best MCTS on the family's DV metric
    # (reach% on antmaze, camping DV-score on maze2d).
    mcss = [x for x in rows if x["method"] == "mcss"]
    mcts = [x for x in rows if x["method"] == "mcts"]
    if mcss and mcts:
        m2 = any(is_maze2d(x["env"]) for x in rows)
        key = (lambda x: x["dv_score"]) if m2 else (lambda x: x["reach"])
        unit, metric = ("", "DV-score") if m2 else (" pp", "reach")
        bm, bt = max(mcss, key=key), max(mcts, key=key)
        print("-" * 96)
        print(f"best MCSS = {key(bm):.1f} ({bm['label']}, {bm['cands']} cand)   "
              f"best MCTS = {key(bt):.1f} ({bt['label']}, {bt['cands']} cand)   "
              f"delta = {key(bt)-key(bm):+.1f}{unit}  [{metric}]")

    print_mcnemar(rows)


if __name__ == "__main__":
    main()
