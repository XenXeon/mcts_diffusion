"""Analyse zero_return_trace_<env>.csv: validate reward cross-checks and
inspect the dist-to-goal trajectory around the closest approach for each
return=0 episode. Pure stdlib — run anywhere the trace CSV exists."""
import csv, sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "results/phase5/zero_return_trace_umaze.csv"
rows = list(csv.DictReader(open(path)))
eps = defaultdict(list)
for r in rows:
    eps[(r["cidx"], r["mode"], r["seed"])].append(r)

# ── 1. Reward cross-checks (validates goal location + 0.5 radius + latch) ──
have_rew = "rew" in rows[0]
print("="*70); print("CROSS-CHECKS"); print("="*70)
if have_rew:
    mism = [r for r in rows if (float(r["rew"])==1.0) != (int(r["in_goal_zone"])==1)]
    print(f"rew==1  vs  in_goal_zone==1  mismatches: {len(mism)}/{len(rows)}"
          f"  ({'OK: goal+radius consistent' if not mism else 'MISMATCH -> goal/radius off!'})")
    for (c,m,s), tr in sorted(eps.items()):
        lat = [float(x["latched_return"]) for x in tr]
        nondec = all(b>=a for a,b in zip(lat,lat[1:]))
        print(f"  cidx={c} {m:<7} seed={s}: latched_return final={lat[-1]:.0f}  "
              f"monotonic={nondec}")
else:
    print("(trace has no 'rew'/'latched_return' columns — re-run the updated script "
          "to enable the reward cross-check)")

# ── 2. Closest-approach inspection for every return=0 episode ──
print("\n"+"="*70); print("CLOSEST-APPROACH TRAJECTORY (return=0 episodes)"); print("="*70)
for (c,m,s), tr in sorted(eps.items()):
    da = [float(x["dist_after"]) for x in tr]
    if min(da) < 0.5:   # reached the zone -> not a zero
        continue
    imin = min(range(len(da)), key=lambda i: da[i])
    near = sum(1 for d in da if 0.5 <= d < 1.0)         # steps lingering just outside
    lo, hi = max(0,imin-10), min(len(da),imin+11)
    # behaviour after closest approach
    after = da[imin:hi]
    leaves = after[-1] - after[0] if len(after)>1 else 0.0
    xy = (tr[imin]["x"], tr[imin]["y"])
    print(f"\ncidx={c} {m:<7} seed={s}  | min_dist={da[imin]:.3f} @ t={imin}  "
          f"pos@min=({xy[0]},{xy[1]})  steps in [0.5,1.0)={near}/{len(da)}  "
          f"post-min Δdist={leaves:+.3f}")
    win = "  ".join(f"{da[i]:.2f}" for i in range(lo,hi))
    print(f"  dist_after [t={lo}..{hi-1}]: {win}")
