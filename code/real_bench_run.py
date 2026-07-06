"""
Run the paper's two central findings on the REAL-geography benchmark, to show they
are not artifacts of synthetic Euclidean instances:

  (A) rho non-identifiability (C1): under the distance-only objective, fix an
      optimal routing and min/max total returns over the flow polytope; a positive
      interval width means rho is not identified on the real network too.
  (B) value of integration (C2): exact joint vs exact myopic under the service
      objective; the capacity-driven regime should reappear.

Instances come from real OSM parcel-locker coordinates and real driving distances
(real_benchmark.build_real_instance). Fully reproducible from place names + seeds.
Output: results/real_benchmark.json / .csv
"""
import csv
import json
import os
import statistics as st

import pulp

import real_benchmark as RB
import certify_rho as CR
import voi as V
import matheuristic as H
import voi_scaled as VS

PLACES = ["Piaseczno, Poland", "Pruszkow, Poland", "Legionowo, Poland"]
SIZES = [8, 10, 12]
SEEDS = [0, 1, 2, 3]
TAU = 2
MU_SERVICE = 5
CAP_FACTOR = 0.7      # binding-ish capacity, so both findings are exercised

# NOTE. The exact joint MILP is NOT used here: on these real, asymmetric,
# large-integer-distance instances both CBC and HiGHS return time-limited
# incumbents mislabelled "Optimal" (verified: the myopic routing, evaluated under
# the coupled joint flow, is feasible at obj 613, yet both MILP solvers report an
# "optimal" >= 650). We therefore measure both findings with the validated
# matheuristic + EXACT flow LP, exactly as the paper's scaled VoI study does:
#   - rho interval: fix a cost-minimising routing (matheuristic, mu=0), then
#     min/max total returns over the flow polytope by two exact LPs (certify_rho);
#   - VoI: myopic matheuristic, then warm-start the joint search from it, so
#     VoI = obj_myopic - obj_joint >= 0 by construction.


def rho_interval(inst, seed=0):
    """rho identified set at a distance-only cost-minimising routing: fix the routing,
    then min/max total returns over the flow polytope (two exact LPs)."""
    r = H.matheuristic(inst, 0, iters=400, seed=seed)     # routing only, distance-only
    if not r["feasible"]:
        return None
    ci = dict(inst); ci["K"] = list(range(1, inst["n"] + 1)); ci["routes"] = r["routes"]
    lo = CR.solve_extreme(ci, pulp.LpMinimize)   # min returns -> max delivery
    hi = CR.solve_extreme(ci, pulp.LpMaximize)   # max returns -> min delivery
    if lo["status"] != "Optimal" or hi["status"] != "Optimal":
        return None
    return {"rho_lo": lo["rho"], "rho_hi": hi["rho"],
            "width": round(hi["rho"] - lo["rho"], 4),
            "R_lo": lo["total_returned_R"], "R_hi": hi["total_returned_R"]}


def voi(inst, mu, seed=0):
    om, myopic_routes = VS.myopic_matheuristic(inst, mu, seed)
    if om is None:
        return None
    j = H.matheuristic(inst, mu, iters=600, seed=seed, init_routes=myopic_routes)
    if not j["feasible"]:
        return None
    v = max(0.0, om - j["objective"])
    return {"VoI": round(v, 2), "VoI_pct": round(100 * v / om, 2) if om else 0.0}


def main():
    rows = []
    for place in PLACES:
        for n in SIZES:
            for seed in SEEDS:
                try:
                    inst = RB.build_real_instance(place, n, TAU, seed)
                except Exception as e:
                    print(f"skip {place} n={n} s={seed}: {type(e).__name__} {e}")
                    continue
                inst["Q"] = max(10, int(CAP_FACTOR * V.total_flow(inst)))
                ri = rho_interval(inst, seed)
                vi = voi(inst, MU_SERVICE, seed)
                if ri is None:
                    continue
                row = {"place": place.split(",")[0], "n": n, "seed": seed,
                       "Q": inst["Q"], "graph_nodes": inst["meta"]["n_graph_nodes"],
                       "rho_lo": ri["rho_lo"], "rho_hi": ri["rho_hi"],
                       "rho_width": ri["width"], "R_lo": ri["R_lo"], "R_hi": ri["R_hi"],
                       "VoI_pct": vi["VoI_pct"] if vi else None}
                rows.append(row)
                print(f"{row['place']:<12} n={n} s={seed}: "
                      f"rho in [{ri['rho_lo']:.3f},{ri['rho_hi']:.3f}] w={ri['width']:.3f}  "
                      f"VoI={row['VoI_pct']}%", flush=True)

    widths = [r["rho_width"] for r in rows]
    vois = [r["VoI_pct"] for r in rows if r["VoI_pct"] is not None]
    summary = {
        "n_instances": len(rows),
        "places": sorted(set(r["place"] for r in rows)),
        "rho_nonident": {
            "frac_width_positive": round(sum(w > 1e-6 for w in widths) / len(widths), 2) if widths else 0,
            "mean_width": round(st.mean(widths), 4) if widths else 0,
            "max_width": round(max(widths), 4) if widths else 0,
        },
        "voi": {
            "n": len(vois),
            "frac_positive": round(sum(v > 1e-6 for v in vois) / len(vois), 2) if vois else 0,
            "mean_pct": round(st.mean(vois), 2) if vois else 0,
            "max_pct": round(max(vois), 2) if vois else 0,
        },
    }
    outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    with open(os.path.join(outdir, "real_benchmark.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    with open(os.path.join(outdir, "real_benchmark.json"), "w") as f:
        json.dump({"config": {"places": PLACES, "sizes": SIZES, "seeds": SEEDS,
                              "tau": TAU, "mu_service": MU_SERVICE, "cap_factor": CAP_FACTOR,
                              "distances": "OSM shortest-path driving, 100 m units"},
                   "summary": summary, "rows": rows}, f, indent=2)
    print("\n=== real-geography benchmark summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
