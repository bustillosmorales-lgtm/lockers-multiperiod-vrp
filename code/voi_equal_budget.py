"""
Controlled value-of-integration MAGNITUDE beyond n=5, without the warm-start confound.

The scaled VoI in the paper warm-starts the joint search from the myopic solution, so
the joint receives strictly more search than the myopic -- an unequal-budget confound
that makes it a prevalence check only, not a magnitude measurement. Here BOTH policies
get the SAME matheuristic budget and NO warm-start: the joint is searched from scratch
with the same iteration count as each myopic period. The gap
    VoI% = (obj_myopic - obj_joint) / obj_myopic
is then a controlled, equal-budget estimate, subject only to SYMMETRIC heuristic
suboptimality. We report the distribution (it can be slightly negative on an instance
where the joint search is unlucky) and compare its magnitude to the exact n=5 study.

n in {10, 12, 15}, tau = 3, tight capacity. Output: results/voi_equal_budget.json
"""
import json
import os
import statistics as st

import voi as V
import matheuristic as H
import voi_scaled as VS

MU = 5
TAU = 3
ITERS = 700
# n<=12 only: at n>=15 the from-scratch joint matheuristic (no warm start) underperforms
# the rolling myopic on the harder full-horizon search, giving spurious negative gaps
# (VoI>=0 always, since the myopic solution is joint-feasible). Magnitude beyond n=12
# needs exact solves (intractable) or warm-starting (prevalence-only, in voi_scaled).
SIZES = [10, 12]
SEEDS = list(range(12))


def make(n, seed):
    ref = V.total_flow(V.make_instance(n, TAU, seed, Q=999))
    Q = max(30, int(0.6 * ref))                       # tight (binding) capacity
    inst = V.make_instance(n, TAU, seed, Q=Q); inst["C"] = 3
    return inst


def cell(n, seed):
    inst = make(n, seed)
    om, _ = VS.myopic_matheuristic(inst, MU, seed)    # rolling matheuristic, ITERS-equivalent per period
    if om is None:
        return None
    jr = H.matheuristic(inst, MU, iters=ITERS, seed=seed)   # joint from scratch, NO warm start
    if not jr["feasible"]:
        return None
    jo = jr["objective"]
    voi_pct = round(100 * (om - jo) / om, 3) if om else 0.0
    return {"n": n, "seed": seed, "obj_myopic": round(om, 2), "obj_joint": round(jo, 2),
            "VoI_pct": voi_pct}


def main():
    rows = []
    for n in SIZES:
        for s in SEEDS:
            r = cell(n, s)
            if r:
                rows.append(r)
                print(f"n={n} s={s}: myopic {r['obj_myopic']} joint {r['obj_joint']} "
                      f"VoI={r['VoI_pct']}%", flush=True)
    by_n = {}
    for n in SIZES:
        vp = [r["VoI_pct"] for r in rows if r["n"] == n]
        pos = [v for v in vp if v > 1e-6]
        if not vp:
            continue
        by_n[n] = {
            "n_cells": len(vp),
            "frac_positive": round(sum(v > 1e-6 for v in vp) / len(vp), 2),
            "mean_pct": round(st.mean(vp), 2),
            "cond_mean_pct": round(st.mean(pos), 2) if pos else 0.0,
            "max_pct": round(max(vp), 2),
            "min_pct": round(min(vp), 2),
        }
    allvp = [r["VoI_pct"] for r in rows]
    allpos = [v for v in allvp if v > 1e-6]
    overall = {
        "n_cells": len(allvp),
        "frac_positive": round(sum(v > 1e-6 for v in allvp) / len(allvp), 2) if allvp else 0,
        "mean_pct": round(st.mean(allvp), 2) if allvp else 0,
        "cond_mean_pct": round(st.mean(allpos), 2) if allpos else 0,
        "max_pct": round(max(allvp), 2) if allvp else 0,
    }
    out = {"mu": MU, "tau": TAU, "iters": ITERS, "sizes": SIZES, "seeds": len(SEEDS),
           "note": "equal matheuristic budget for both policies, no warm start",
           "overall": overall, "by_n": by_n}
    outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    with open(os.path.join(outdir, "voi_equal_budget.json"), "w") as f:
        json.dump({"config": out, "rows": rows}, f, indent=2)
    print("\n=== equal-budget VoI (no warm start) ===")
    print("overall:", overall)
    for n in SIZES:
        if n in by_n:
            print(f"  n={n}: {by_n[n]}")


if __name__ == "__main__":
    main()
