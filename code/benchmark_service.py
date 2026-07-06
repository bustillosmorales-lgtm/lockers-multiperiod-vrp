"""
Regenerated benchmark under the SERVICE-AWARE objective, to report the return-flow
ratio as an IDENTIFIED quantity (contrast with Section 6, where it is not).

The original 48-instance benchmark of the base paper is not published, so we
regenerate a comparable set following the same design: eight network sizes
{5,8,10,12,15,20,25,30} x six scenarios {low demand, base, high demand, reduced
fleet, reduced vehicle capacity, reduced locker capacity}, one instance per cell.
Each instance is solved under min dist + mu*R with mu=5 by the matheuristic
(0% gap vs the exact optimum where verifiable), which minimizes recirculation, so
the reported rho = R / (Y + R) is the identified operational value under that
objective (higher mu lowers it further).

Run:  python benchmark_service.py
"""

import csv
import json
import math
import os
import random
import statistics as st

import matheuristic as H
from flow_eval_fast import eval_flow_fast

MU = 5
TAU = 2
SIZES = [5, 8, 10, 12, 15, 20, 25, 30]
SCENARIOS = ["low demand", "base", "high demand",
             "reduced fleet", "reduced vehicle capacity", "reduced locker capacity"]


def base_instance(n, seed, dem_factor=1.0):
    rng = random.Random(seed)
    coords = {0: (50, 50)}
    for k in range(1, n + 1):
        coords[k] = (rng.uniform(0, 100), rng.uniform(0, 100))
    d = {(i, j): round(math.dist(coords[i], coords[j]))
         for i in coords for j in coords if i != j}
    T = list(range(1, TAU + 1))
    p = {t: {j: {k: (round(rng.randint(0, 3) * dem_factor) if j != k else 0)
                 for k in range(1, n + 1)} for j in range(1, n + 1)} for t in T}
    g = {t: {k: round(rng.randint(0, 3) * dem_factor) for k in range(1, n + 1)} for t in T}
    return coords, d, T, p, g


def make_bench(n, scenario, seed):
    dem = {"low demand": 0.6, "high demand": 1.4}.get(scenario, 1.0)
    coords, d, T, p, g = base_instance(n, seed, dem)
    ref = sum(g[1].values()) + sum(p[1][j][k] for j in p[1] for k in p[1][j])
    C = max(2, round(n / 5) + 1)
    Q = max(20, int(0.6 * ref))
    H_ = 30
    if scenario == "reduced fleet":
        C = max(2, C - 1)
    elif scenario == "reduced vehicle capacity":
        Q = max(14, int(0.7 * Q))
    elif scenario == "reduced locker capacity":
        H_ = 15
    return {"n": n, "T": T, "d": d, "p": p, "g": g, "C": C, "Q": Q, "H": H_}


def solve_cell(n, scenario, seed):
    """Solve one cell under the service objective; loosen Q if infeasible."""
    inst = make_bench(n, scenario, seed)
    for attempt in range(4):
        res = H.matheuristic(inst, MU, iters=500, seed=seed)
        if res["feasible"]:
            det = eval_flow_fast(inst, res["routes"], MU, return_detail=True)
            Y = sum(v for per in det["y_by_period"].values() for v in per.values())
            R = res["R"]
            rho = R / (Y + R) if (Y + R) > 0 else 0.0
            veh = max(len(res["routes"][t]) for t in inst["T"])
            return {"n": n, "scenario": scenario, "obj": res["objective"],
                    "R": R, "Y": Y, "rho": round(rho, 4), "vehicles": veh,
                    "Q": inst["Q"], "C": inst["C"], "H": inst["H"]}
        inst["Q"] = int(inst["Q"] * 1.25)   # loosen and retry
    return None


def main():
    rows = []
    for n in SIZES:
        for si, sc in enumerate(SCENARIOS):
            r = solve_cell(n, sc, seed=n * 10 + si)
            if r:
                rows.append(r)
                print(f"n={n:>2} {sc:<25} rho={r['rho']:.4f} obj={r['obj']} veh={r['vehicles']}")

    def agg(key, val):
        groups = {}
        for r in rows:
            groups.setdefault(r[key], []).append(r)
        out = {}
        for g, rs in groups.items():
            out[g] = {"n_cells": len(rs),
                      "avg_obj": round(st.mean(x["obj"] for x in rs), 1),
                      "avg_vehicles": round(st.mean(x["vehicles"] for x in rs), 2),
                      "avg_rho": round(st.mean(x["rho"] for x in rs), 4)}
        return out

    by_size = agg("n", None)
    by_scenario = agg("scenario", None)

    outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    with open(os.path.join(outdir, "benchmark_service.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    with open(os.path.join(outdir, "benchmark_service.json"), "w") as f:
        json.dump({"mu": MU, "tau": TAU, "by_size": by_size,
                   "by_scenario": by_scenario}, f, indent=2)

    print("\n=== identified rho by size (service objective, mu=5) ===")
    for n in SIZES:
        if n in by_size:
            s = by_size[n]
            print(f"  {n:>2}: obj={s['avg_obj']:>7} veh={s['avg_vehicles']:>4} rho={s['avg_rho']:.4f}")
    print("=== identified rho by scenario ===")
    for sc in SCENARIOS:
        if sc in by_scenario:
            s = by_scenario[sc]
            print(f"  {sc:<25}: obj={s['avg_obj']:>7} veh={s['avg_vehicles']:>4} rho={s['avg_rho']:.4f}")


if __name__ == "__main__":
    main()
