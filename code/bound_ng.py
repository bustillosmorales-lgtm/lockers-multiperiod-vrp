"""
Validate the ng-route CG bound: (a) VALID -- ng LB <= DP LB <= matheuristic UB, and
ng LB well above the arc-based LP; (b) USEFUL -- ng runs and gives a bound past the
n<=15 ceiling where the bitmask DP is intractable. mu = service objective (so the
CG bound has a non-trivial flow-relaxation gap to close).

Output: results/bound_ng.json / printed table.
"""
import json
import os
import time

import voi as V
import model_full as M
import branch_price as BP

MU = 5
TAU = 2


def make(n, seed=1):
    Q = max(30, int(1.2 * V.total_flow(V.make_instance(n, TAU, seed, Q=999))))
    inst = V.make_instance(n, TAU, seed, Q=Q); inst["C"] = 3
    return inst


def run():
    rows = []
    # small: DP feasible -> check ng <= dp <= UB and both >> LP
    for n in [10, 12, 15]:
        inst = make(n)
        lp = M.lower_bound(inst, MU, time_limit=60)
        t0 = time.time(); ub = None
        import matheuristic as H
        ub = H.matheuristic(inst, MU, iters=500, seed=1).get("objective")
        t_ub = time.time() - t0
        t0 = time.time(); dp = BP.cg_lower_bound(inst, MU, pricer="dp") if n <= 15 else None
        t_dp = time.time() - t0
        t0 = time.time(); ng = BP.cg_lower_bound(inst, MU, pricer="ng", ng_size=7); t_ng = time.time() - t0
        row = {"n": n, "LP": lp, "CG_dp": dp, "CG_ng": ng, "UB_heur": ub,
               "t_dp_s": round(t_dp, 1), "t_ng_s": round(t_ng, 1),
               "gap_ng_pct": round(100 * (ub - ng) / ub, 1) if (ub and ng) else None,
               "valid_ng_le_dp": (ng <= dp + 1e-6) if (dp is not None and ng is not None) else None,
               "valid_ng_le_ub": (ng <= ub + 1e-6) if (ub and ng) else None,
               "ng_above_lp": (ng >= lp - 1e-6) if (ng is not None and lp is not None) else None}
        rows.append(row); print(row, flush=True)

    # large: DP intractable (2^n) -> ng-only bound, compare to matheuristic UB
    for n in [18, 20, 22]:
        inst = make(n)
        lp = M.lower_bound(inst, MU, time_limit=90)
        import matheuristic as H
        ub = H.matheuristic(inst, MU, iters=500, seed=1).get("objective")
        t0 = time.time(); ng = BP.cg_lower_bound(inst, MU, pricer="ng", ng_size=7); t_ng = time.time() - t0
        row = {"n": n, "LP": lp, "CG_dp": None, "CG_ng": ng, "UB_heur": ub,
               "t_dp_s": None, "t_ng_s": round(t_ng, 1),
               "gap_ng_pct": round(100 * (ub - ng) / ub, 1) if (ub and ng) else None,
               "valid_ng_le_dp": None,
               "valid_ng_le_ub": (ng <= ub + 1e-6) if (ub and ng) else None,
               "ng_above_lp": (ng >= lp - 1e-6) if (ng is not None and lp is not None) else None}
        rows.append(row); print(row, flush=True)

    outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    with open(os.path.join(outdir, "bound_ng.json"), "w") as f:
        json.dump({"mu": MU, "tau": TAU, "ng_size": 7, "rows": rows}, f, indent=2)
    print("\nwritten bound_ng.json")


if __name__ == "__main__":
    run()
