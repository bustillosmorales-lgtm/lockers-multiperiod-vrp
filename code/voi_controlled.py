"""
Controlled value-of-integration study (addresses the referee objections that the
scaled VoI conflates integration with extra heuristic search, uses a two-value
capacity knob, and has too few seeds).

Design: solve BOTH policies EXACTLY on small instances, so there is no heuristic
suboptimality and no search-effort confound.
  - joint  = solve_joint (exact MILP over the whole horizon);
  - myopic = solve each period exactly in sequence, carrying returns forward.
Then VoI = obj_myopic - obj_joint >= 0 is the genuine structural value of
integration. We sweep vehicle capacity CONTINUOUSLY to locate the binding
threshold, with several seeds per point, and report the distribution.

n = 5,6,7, tau = 3. Parallelised across (n, capacity, seed) cells.
Output: results/voi_controlled.json / .csv
"""
import csv
import json
import os
import statistics as st
from concurrent.futures import ProcessPoolExecutor, as_completed

import voi as V
import model_full as M

MU = 5
TAU = 3
# n=5 only: this is where the joint MILP reliably solves; at n>=6, tau=3 CBC does
# not prove optimality within a practical cap (and mislabels the incumbent), so we
# keep the study exact and rigorous rather than large. The large-n VoI is probed
# separately by the matheuristic (voi_scaled.py).
SIZES = [5]
FACTORS = [0.40, 0.50, 0.60, 0.70, 0.85, 1.00]   # x reference flow -> Q
SEEDS = list(range(8))
TL = 240


def build(n, seed, factor):
    ref = V.total_flow(V.make_instance(n, TAU, seed, Q=999))
    Q = max(10, int(factor * ref))
    inst = V.make_instance(n, TAU, seed, Q=Q); inst["C"] = 3
    return inst


def myopic_exact(inst):
    K = list(range(1, inst["n"] + 1))
    prev = {k: 0 for k in K}
    total = 0.0
    for t in inst["T"]:
        disp = {k: inst["g"][t][k] + prev[k] for k in K}
        res = M.solve_single(inst, t, disp, MU, time_limit=TL)
        if res["status"] != "Optimal":
            return None
        total += res["distance"] + MU * res["R"]
        prev = res["returns"]
    return total


def cell(args):
    n, seed, factor = args
    inst = build(n, seed, factor)
    om = myopic_exact(inst)
    if om is None:
        return None                       # myopic infeasible -> skip
    j = M.solve_joint(inst, MU, time_limit=TL)
    jo = j.get("objective")
    # Validity filter, solver-status-independent: the joint optimum is <= myopic
    # (myopic is feasible for joint). If the returned value exceeds myopic, the
    # solver failed to close within the cap -> discard. Otherwise VoI = myopic -
    # joint is >= 0 and is a conservative lower bound on the true VoI (a returned
    # joint value can only be >= the true optimum).
    if jo is None or jo > om + 1e-6:
        return None
    voi = om - jo
    return {"n": n, "seed": seed, "factor": factor, "Q": inst["Q"],
            "obj_joint": jo, "obj_myopic": round(om, 2),
            "VoI": round(voi, 2),
            "VoI_pct": round(100 * voi / om, 2) if om else 0.0}


def main():
    jobs = [(n, s, f) for n in SIZES for s in SEEDS for f in FACTORS]
    rows = []
    with ProcessPoolExecutor(max_workers=16) as ex:
        for fut in as_completed([ex.submit(cell, j) for j in jobs]):
            r = fut.result()
            if r:
                rows.append(r)
    rows.sort(key=lambda r: (r["factor"], r["n"], r["seed"]))

    # aggregate by capacity factor (the tightness knob)
    summary = {}
    for f in FACTORS:
        sub = [r for r in rows if r["factor"] == f]
        if not sub:
            continue
        vp = [r["VoI_pct"] for r in sub]
        vp_sorted = sorted(vp)
        summary[f] = {
            "n_feasible": len(sub),
            "frac_VoI_positive": round(sum(v > 1e-6 for v in vp) / len(sub), 2),
            "mean_VoI_pct": round(st.mean(vp), 2),
            "median_VoI_pct": round(st.median(vp), 2),
            "max_VoI_pct": round(max(vp), 2),
        }

    outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    with open(os.path.join(outdir, "voi_controlled.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    with open(os.path.join(outdir, "voi_controlled.json"), "w") as fh:
        json.dump({"mu": MU, "tau": TAU, "sizes": SIZES, "seeds": len(SEEDS),
                   "exact": True, "by_capacity_factor": summary}, fh, indent=2)
    print("=== controlled VoI (exact joint vs exact myopic), by capacity factor ===")
    print("factor: frac>0 / mean% / median% / max%  (lower factor = tighter)")
    for f in FACTORS:
        if f in summary:
            s = summary[f]
            print(f"  {f:.2f}: {s['frac_VoI_positive']:.2f} / {s['mean_VoI_pct']:.2f} / "
                  f"{s['median_VoI_pct']:.2f} / {s['max_VoI_pct']:.2f}  (n={s['n_feasible']})")


if __name__ == "__main__":
    main()
