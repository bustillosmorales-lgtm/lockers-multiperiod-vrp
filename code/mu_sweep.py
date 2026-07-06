"""
Sweep the deferral penalty mu to trace the return-flow ratio rho and the routing
distance as functions of the service weight. Shows that under the service-aware
objective rho is a tunable operating point: mu=0 leaves it non-identified (an
interval), and increasing mu trades routing distance for lower recirculation.

Output: results/mu_sweep.json  (rho(mu), distance(mu), plus the mu=0 interval).
"""

import json
import os
import statistics as st

import voi as V
import matheuristic as H
from flow_eval_fast import eval_flow_fast

N = 10
TAU = 2
SEEDS = list(range(5))
MUS = [0, 0.5, 1, 2, 3, 5, 8, 12, 20]


def rho_of(inst, routes, mu):
    det = eval_flow_fast(inst, routes, mu, return_detail=True)
    Y = sum(v for per in det["y_by_period"].values() for v in per.values())
    R = det["R"]
    return (R / (Y + R) if (Y + R) > 0 else 0.0), det["distance"], R


def main():
    per_mu = {mu: [] for mu in MUS}
    dist_mu = {mu: [] for mu in MUS}
    interval0 = []                       # (rho_min, rho_max) at mu=0
    for seed in SEEDS:
        inst = V.make_instance(N, TAU, seed, Q=max(20, int(0.55 * V.total_flow(
            V.make_instance(N, TAU, seed, Q=999))))); inst["C"] = 3
        # min-distance routing (mu=0) to expose the non-identified rho interval
        base = H.matheuristic(inst, 0, iters=400, seed=seed)
        if not base["feasible"]:
            continue
        r_route = base["routes"]
        rmin = eval_flow_fast(inst, r_route, 1.0, return_detail=True)   # minimizes R
        rmax = eval_flow_fast(inst, r_route, -1.0, return_detail=True)  # maximizes R
        def rho_from(det):
            Y = sum(v for per in det["y_by_period"].values() for v in per.values())
            return det["R"] / (Y + det["R"]) if (Y + det["R"]) > 0 else 0.0
        interval0.append((rho_from(rmin), rho_from(rmax)))
        # sweep mu (re-optimize routing at each mu)
        for mu in MUS:
            sol = H.matheuristic(inst, mu, iters=400, seed=seed)
            if not sol["feasible"]:
                continue
            rho, dist, R = rho_of(inst, sol["routes"], mu)
            per_mu[mu].append(rho)
            dist_mu[mu].append(dist)

    out = {
        "n": N, "tau": TAU, "seeds": len(SEEDS),
        "mu_grid": MUS,
        "rho_mean": {mu: round(st.mean(v), 4) for mu, v in per_mu.items() if v},
        "dist_mean": {mu: round(st.mean(v), 1) for mu, v in dist_mu.items() if v},
        "rho0_interval_mean": [round(st.mean(a for a, _ in interval0), 4),
                               round(st.mean(b for _, b in interval0), 4)]
        if interval0 else None,
    }
    outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    with open(os.path.join(outdir, "mu_sweep.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
