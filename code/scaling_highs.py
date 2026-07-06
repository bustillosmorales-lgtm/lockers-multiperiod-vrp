"""
Honest scaling comparison against a STRONG open-source solver (HiGHS), not CBC.
This replaces the CBC baseline (which was weak and let the matheuristic look
dominant). HiGHS is Gurobi-class open source. The comparison reveals a
speed/quality/robustness trade-off, not dominance:
  - small n: HiGHS finds better solutions than the matheuristic, but far slower;
  - large n: HiGHS fails to return a usable incumbent, while the matheuristic does.

Output: results/scaling_highs.json / .csv
"""
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import voi as V
import model_full as M
import matheuristic as H

MU = 5
TAU = 2
HIGHS_TL = 300
SIZES = [5, 8, 10, 12, 15, 20, 25, 30]


def make(n, seed=7):
    ref = V.total_flow(V.make_instance(n, TAU, seed, Q=999))
    Q = max(60, int(1.5 * ref))
    inst = V.make_instance(n, TAU, seed, Q=Q); inst["C"] = 5; inst["H"] = max(40, 5 * n)
    return inst


def run_size(n):
    inst = make(n)
    t0 = time.time(); he = H.matheuristic(inst, MU, iters=600, seed=7); t_he = time.time() - t0
    t0 = time.time(); jh = M.solve_joint(inst, MU, time_limit=HIGHS_TL, solver="highs"); t_hi = time.time() - t0
    ho = he.get("objective") if he.get("feasible") else None
    Ho = jh.get("objective")
    proven = jh.get("status") == "Optimal" and t_hi < HIGHS_TL - 5
    solved = jh.get("status") not in ("Not Solved", "Infeasible", "Undefined") and Ho not in (None, 0.0)
    vs = (round(100 * (ho - Ho) / Ho, 1) if (ho and solved) else None)
    return {"n": n, "heur_obj": ho, "heur_time_s": round(t_he, 1),
            "highs_obj": Ho if solved else None, "highs_time_s": round(t_hi, 1),
            "highs_proven": proven, "highs_solved": solved,
            "heur_vs_highs_%": vs}


def main():
    rows = []
    with ProcessPoolExecutor(max_workers=len(SIZES)) as ex:
        for fut in as_completed([ex.submit(run_size, n) for n in SIZES]):
            r = fut.result(); rows.append(r)
            tag = ("proven" if r["highs_proven"] else "incumbent" if r["highs_solved"]
                   else "NO SOLUTION")
            print(f"n={r['n']:>2} heur={r['heur_obj']} ({r['heur_time_s']}s)  "
                  f"HiGHS={r['highs_obj']} [{tag} {r['highs_time_s']:.0f}s]  "
                  f"heur vs HiGHS={r['heur_vs_highs_%']}%", flush=True)
    rows.sort(key=lambda r: r["n"])
    outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    with open(os.path.join(outdir, "scaling_highs.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    with open(os.path.join(outdir, "scaling_highs.json"), "w") as f:
        json.dump({"mu": MU, "tau": TAU, "solver": "HiGHS", "highs_time_limit_s": HIGHS_TL,
                   "rows": rows}, f, indent=2)
    print("\nwritten scaling_highs.json")


if __name__ == "__main__":
    main()
