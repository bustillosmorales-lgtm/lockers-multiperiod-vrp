"""
Value-of-integration (VoI) experiment: joint horizon vs myopic rolling horizon.

VoI is the optimality gap of the myopic (day-by-day, no lookahead) policy
against the joint (whole-horizon) optimum, under one global objective
    obj = total_distance + mu * total_returns.
Because any myopic solution is feasible for the joint problem, obj_joint <=
obj_myopic, so
    VoI = obj_myopic - obj_joint >= 0        (theoretical guarantee)
and the empirical question is WHEN it is large. We sweep two controls:
  - mu       : 0 = paper's distance-only objective; >0 = service-aware.
  - capacity : loose (non-binding) vs tight (binding), the recirculation coupling.

Expected regimes (cf. the model's structure and the Phase-A result):
  mu=0, loose  -> VoI ~ 0     (integration inert: distance separates by period)
  mu=0, tight  -> VoI > 0 sometimes (capacity couples periods through returns)
  mu>0, loose  -> VoI small
  mu>0, tight  -> VoI largest (joint coordinates deferrals to cut recirculation)

Run:  python voi.py
Deps: pulp (bundled CBC).
"""

import csv
import json
import math
import os
import random
import statistics as st

import model_full as M

JOINT_TL = 25   # joint solve time limit (s); incumbent keeps VoI a lower bound


def make_instance(n, tau, seed, Q, C=2, H=40, dem_hi=3):
    rng = random.Random(seed)
    coords = {0: (50, 50)}
    for k in range(1, n + 1):
        coords[k] = (rng.uniform(0, 100), rng.uniform(0, 100))
    d = {(i, j): round(math.dist(coords[i], coords[j]))
         for i in coords for j in coords if i != j}
    T = list(range(1, tau + 1))
    p = {t: {j: {k: (rng.randint(0, dem_hi) if j != k else 0)
                 for k in range(1, n + 1)} for j in range(1, n + 1)} for t in T}
    g = {t: {k: rng.randint(0, dem_hi) for k in range(1, n + 1)} for t in T}
    return {"n": n, "T": T, "d": d, "p": p, "g": g, "C": C, "Q": Q, "H": H}


def total_flow(inst):
    """Reference flow scale: period-1 depot dispatch + all locker pickups."""
    T0 = inst["T"][0]
    disp = sum(inst["g"][T0].values())
    pick = sum(inst["p"][T0][j][k] for j in inst["p"][T0] for k in inst["p"][T0][j])
    return disp + pick


def obj_of(sol, mu):
    return sol["total_distance"] + mu * sol["total_R"]


def run_config(n, tau, seed, Q, mu):
    inst = make_instance(n, tau, seed, Q)
    j = M.solve_joint(inst, mu, time_limit=JOINT_TL)
    if j["status"] == "Infeasible":
        return None
    m = M.solve_myopic(inst, mu, time_limit=60)
    oj, om = obj_of(j, mu), obj_of(m, mu)
    voi = om - oj
    return {
        "n": n, "tau": tau, "seed": seed, "Q": Q, "mu": mu,
        "obj_joint": round(oj, 2), "obj_myopic": round(om, 2),
        "VoI": round(voi, 2), "VoI_pct": round(100 * voi / oj, 2) if oj else 0.0,
        "dist_joint": j["total_distance"], "dist_myopic": m["total_distance"],
        "R_joint": j["total_R"], "R_myopic": m["total_R"],
        "R_reduction": m["total_R"] - j["total_R"],
        "joint_status": j["status"],
    }


def main():
    n, tau = 5, 2
    seeds = list(range(12))
    # capacity regimes as multiples of the reference flow scale
    ref = total_flow(make_instance(n, tau, 0, Q=999))
    regimes = {"loose": max(30, int(1.6 * ref)), "tight": max(14, int(0.6 * ref))}
    mus = [0, 5]

    rows = []
    for rlabel, Q in regimes.items():
        for mu in mus:
            for s in seeds:
                r = run_config(n, tau, s, Q, mu)
                if r is None:
                    continue
                r["regime"] = rlabel
                rows.append(r)
                print(f"[{rlabel} Q={Q} mu={mu} seed={s}] "
                      f"VoI={r['VoI']:>6} ({r['VoI_pct']:>5}%)  "
                      f"R_j={r['R_joint']} R_m={r['R_myopic']}")

    # summary by (regime, mu)
    summary = {}
    for rlabel in regimes:
        for mu in mus:
            sub = [x for x in rows if x["regime"] == rlabel and x["mu"] == mu]
            if not sub:
                continue
            vois = [x["VoI"] for x in sub]
            summary[f"{rlabel}_mu{mu}"] = {
                "Q": regimes[rlabel], "n_instances": len(sub),
                "mean_VoI": round(st.mean(vois), 2),
                "mean_VoI_pct": round(st.mean([x["VoI_pct"] for x in sub]), 2),
                "max_VoI_pct": round(max(x["VoI_pct"] for x in sub), 2),
                "frac_VoI_positive": round(sum(v > 1e-6 for v in vois) / len(sub), 2),
                "mean_R_reduction": round(st.mean([x["R_reduction"] for x in sub]), 2),
            }

    outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "voi_runs.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(outdir, "voi_summary.json"), "w") as f:
        json.dump({"config": {"n": n, "tau": tau, "seeds": len(seeds),
                              "regimes": regimes, "mus": mus,
                              "joint_time_limit_s": JOINT_TL},
                   "summary": summary}, f, indent=2)

    print("\n=== SUMMARY (VoI = obj_myopic - obj_joint, >=0 by construction) ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWritten: {outdir}\\voi_runs.csv  and  voi_summary.json")


if __name__ == "__main__":
    main()
