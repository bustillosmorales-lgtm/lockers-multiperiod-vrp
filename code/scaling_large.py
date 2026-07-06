"""
Large-scale demonstration: the matheuristic at n=40..100, where no exact solver is
usable. The flow evaluation uses only the route arcs (~n), so it is O(n^2) per
call, not O(n^3), and the method keeps producing solutions in reasonable time.
HiGHS is run at n=40,50 (300 s) to confirm it cannot compete at that scale.

Output: results/scaling_large.json
"""
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import voi as V
import matheuristic as H
import model_full as M

MU = 5
TAU = 2
SIZES = [40, 50, 75, 100]
HIGHS_SIZES = [40, 50]
HIGHS_TL = 300


def make(n, seed=7):
    ref = V.total_flow(V.make_instance(n, TAU, seed, Q=999))
    Q = max(60, int(1.5 * ref))
    inst = V.make_instance(n, TAU, seed, Q=Q); inst["C"] = max(5, n // 8); inst["H"] = 5 * n
    return inst


def run_size(n):
    inst = make(n)
    t0 = time.time(); he = H.matheuristic(inst, MU, iters=500, seed=7); t_he = time.time() - t0
    row = {"n": n, "heur_obj": he.get("objective") if he.get("feasible") else None,
           "heur_time_s": round(t_he, 1), "highs_obj": None, "highs_time_s": None}
    if n in HIGHS_SIZES:
        t0 = time.time(); jh = M.solve_joint(inst, MU, time_limit=HIGHS_TL, solver="highs")
        row["highs_time_s"] = round(time.time() - t0, 1)
        Ho = jh.get("objective")
        row["highs_obj"] = Ho if jh.get("status") not in ("Not Solved", "Infeasible") and Ho not in (None, 0.0) else None
    return row


def main():
    rows = []
    with ProcessPoolExecutor(max_workers=len(SIZES)) as ex:
        for fut in as_completed([ex.submit(run_size, n) for n in SIZES]):
            r = fut.result(); rows.append(r)
            hi = f"HiGHS={r['highs_obj']} ({r['highs_time_s']}s)" if r["highs_time_s"] else ""
            print(f"n={r['n']:>3} heur={r['heur_obj']} ({r['heur_time_s']}s)  {hi}", flush=True)
    rows.sort(key=lambda r: r["n"])
    outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    with open(os.path.join(outdir, "scaling_large.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    with open(os.path.join(outdir, "scaling_large.json"), "w") as f:
        json.dump({"mu": MU, "tau": TAU, "rows": rows}, f, indent=2)
    print("\nwritten scaling_large.json")


if __name__ == "__main__":
    main()
