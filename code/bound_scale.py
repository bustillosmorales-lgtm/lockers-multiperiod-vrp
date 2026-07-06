"""
Optimality bounds at the matheuristic's operating range, using the DP pricer.

At n=12-15 the exact optimum is unknown, but the column-generation (root) bound
with the return valid inequalities is a valid LOWER bound, and the matheuristic
gives an UPPER bound. Their ratio is a genuine optimality gap for the matheuristic
at sizes the MILP pricer (and the exact solver) cannot reach. This extends the
certified range from n<=8 to n<=15.

Output: results/bound_scale.json
Run:  python bound_scale.py
"""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import voi as V
import matheuristic as H
import branch_price as BP

MU = 5
SIZES = [10, 12, 15]


def make(n, seed=1):
    Q = max(30, int(1.2 * V.total_flow(V.make_instance(n, 2, seed, Q=999))))
    inst = V.make_instance(n, 2, seed, Q=Q); inst["C"] = 3
    return inst


def run_size(n):
    inst = make(n)
    t0 = time.time(); he = H.matheuristic(inst, MU, iters=600, seed=7); t_he = time.time() - t0
    t0 = time.time(); lb = BP.cg_lower_bound(inst, MU, max_iter=60,
                                             return_cuts="3cycle", pricer="dp")
    t_lb = time.time() - t0
    gap = round(100 * (he["objective"] - lb) / he["objective"], 1) if he["objective"] else None
    return {"n": n, "matheuristic_obj": he["objective"], "matheuristic_time_s": round(t_he, 1),
            "cg_lower_bound": lb, "cg_time_s": round(t_lb, 1), "gap_%": gap}


def main():
    rows = []
    with ProcessPoolExecutor(max_workers=len(SIZES)) as ex:
        for fut in as_completed([ex.submit(run_size, n) for n in SIZES]):
            r = fut.result(); rows.append(r)
            print(f"n={r['n']:>2}  heur={r['matheuristic_obj']}  CG-LB={r['cg_lower_bound']}  "
                  f"gap={r['gap_%']}%  (cg {r['cg_time_s']}s)", flush=True)
    rows.sort(key=lambda r: r["n"])
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "bound_scale.json"))
    with open(out, "w") as f:
        json.dump({"mu": MU, "pricer": "dp", "rows": rows}, f, indent=2)
    print("\nwritten bound_scale.json")


if __name__ == "__main__":
    main()
