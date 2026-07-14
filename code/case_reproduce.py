"""
Reproduce the Kandoo Guayaquil case study with the open-source pipeline
(CBC / HiGHS) and check it against the manuscript's Gurobi result (obj 270,
gap 0). Writes results/case_reproduce.json.
"""
import json
import os
import time

import david_data as D
import model_full as M

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))


def route_from(sol):
    out = {}
    for t, r in sol["per_period"].items():
        out[t] = M._routes_from_arcs(r["used_arcs"], 15)
    return out


def main(solver="highs", tl=1200):
    inst = D.build_case_study()

    t0 = time.time()
    lb = M.lower_bound(inst, mu=0, time_limit=120)
    t_lb = time.time() - t0

    t0 = time.time()
    sol = M.solve_joint(inst, mu=0, time_limit=tl, solver=solver)
    t_solve = time.time() - t0

    problems = M.validate(inst, sol)
    res = {
        "solver": solver, "time_limit_s": tl,
        "lp_lower_bound": lb, "lp_lb_time_s": round(t_lb, 1),
        "status": sol["status"],
        "objective": sol["objective"],
        "total_distance": sol["total_distance"],
        "total_R": sol["total_R"],
        "solve_time_s": round(t_solve, 1),
        "manuscript_gurobi_obj": 270,
        "matches_manuscript": abs(sol["objective"] - 270) < 1e-6,
        "gap_to_270_pct": round(100 * (sol["objective"] - 270) / 270, 4),
        "validation_problems": problems,
        "routes": {str(t): rr for t, rr in route_from(sol).items()},
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "case_reproduce.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "routes"}, indent=2))
    print("routes:", res["routes"])


if __name__ == "__main__":
    import sys
    solver = sys.argv[1] if len(sys.argv) > 1 else "highs"
    tl = int(sys.argv[2]) if len(sys.argv) > 2 else 1200
    main(solver, tl)
