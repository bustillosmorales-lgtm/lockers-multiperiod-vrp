"""
Persist the lower-bound tightening ladder to results/bound_ladder.json, so the
gap ranges quoted in the paper are regenerable rather than hard-coded in the
figure script. For each instance: exact objective (CBC, proven flag), and the
gap to it for LP relaxation, +subtour cuts, +column-generation (root),
+2-cycle return inequalities, +3-cycle return inequalities.

Run:  python bound_ladder.py
"""
import time
import json
import os

import voi as V
import model_full as M
import bound_cuts as BC
import branch_price as BP

MU = 5
CASES = [(5, 1), (5, 0), (5, 2), (8, 1), (8, 2)]


def gap(exact, lb):
    return round(100 * (exact - lb) / exact, 1)


def main():
    rows = []
    for n, seed in CASES:
        Q = max(30, int(1.2 * V.total_flow(V.make_instance(n, 2, seed, Q=999))))
        inst = V.make_instance(n, 2, seed, Q); inst["C"] = 3
        TL = 90
        t0 = time.time(); ex = M.solve_joint(inst, MU, time_limit=TL); t_ex = time.time() - t0
        # CBC labels time-limited incumbents "Optimal"; only trust it if the solve
        # finished comfortably inside the cap.
        proven = ex["status"] == "Optimal" and t_ex < 0.9 * TL
        exo = ex["objective"]
        lp = M.lower_bound(inst, MU, time_limit=30)
        cuts = BC.lower_bound_cuts(inst, MU, max_rounds=20, time_limit=20)
        cg = BP.cg_lower_bound(inst, MU, max_iter=50, return_cuts=None)
        cg2 = BP.cg_lower_bound(inst, MU, max_iter=50, return_cuts="2cycle")
        cg3 = BP.cg_lower_bound(inst, MU, max_iter=50, return_cuts="3cycle")
        row = {"n": n, "seed": seed, "exact": exo, "exact_proven": proven,
               "gap_LP": gap(exo, lp), "gap_cuts": gap(exo, cuts),
               "gap_CG": gap(exo, cg), "gap_CG_2cycle": gap(exo, cg2),
               "gap_CG_3cycle": gap(exo, cg3)}
        rows.append(row)
        print(row)

    def rng(key):
        vals = [r[key] for r in rows]
        return [min(vals), max(vals)]

    summary = {
        "mu": MU,
        "range_gap_LP": rng("gap_LP"),
        "range_gap_CG_alone": rng("gap_CG"),
        "range_gap_CG_3cycle": rng("gap_CG_3cycle"),
        "rows": rows,
    }
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "results", "bound_ladder.json"))
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print("\nranges:", {k: v for k, v in summary.items() if k.startswith("range")})
    print("written", out)


if __name__ == "__main__":
    main()
