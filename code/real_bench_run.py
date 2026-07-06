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
import model_full as M
import certify_rho as CR
import voi as V

PLACES = ["Piaseczno, Poland", "Pruszkow, Poland", "Legionowo, Poland"]
SIZES = [8, 10, 12]
SEEDS = [0, 1, 2, 3]
TAU = 2
MU_SERVICE = 5
CAP_FACTOR = 0.7      # binding-ish capacity, so both findings are exercised
TL = 120


def rho_interval(inst):
    """rho identified set at a distance-only optimal routing (fix routing, min/max R)."""
    j = M.solve_joint(inst, mu=0, time_limit=TL)
    if j["status"] not in ("Optimal",):
        return None
    routes = {t: [[i, j2] for (i, j2) in j["per_period"][t]["used_arcs"]] for t in inst["T"]}
    ci = dict(inst); ci["K"] = list(range(1, inst["n"] + 1)); ci["routes"] = routes
    lo = CR.solve_extreme(ci, pulp.LpMinimize)   # min returns -> max delivery
    hi = CR.solve_extreme(ci, pulp.LpMaximize)   # max returns -> min delivery
    if lo["status"] != "Optimal" or hi["status"] != "Optimal":
        return None
    return {"rho_lo": lo["rho"], "rho_hi": hi["rho"],
            "width": round(hi["rho"] - lo["rho"], 4),
            "R_lo": lo["total_returned_R"], "R_hi": hi["total_returned_R"]}


def myopic_exact(inst, mu):
    K = list(range(1, inst["n"] + 1)); prev = {k: 0 for k in K}; total = 0.0
    for t in inst["T"]:
        disp = {k: inst["g"][t][k] + prev[k] for k in K}
        res = M.solve_single(inst, t, disp, mu, time_limit=TL)
        if res["status"] != "Optimal":
            return None
        total += res["distance"] + mu * res["R"]; prev = res["returns"]
    return total


def voi(inst, mu):
    om = myopic_exact(inst, mu)
    if om is None:
        return None
    j = M.solve_joint(inst, mu, time_limit=TL, solver="highs")
    jo = j.get("objective")
    if jo is None or jo > om + 1e-6:
        return None
    return {"VoI": round(om - jo, 2), "VoI_pct": round(100 * (om - jo) / om, 2) if om else 0.0}


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
                ri = rho_interval(inst)
                vi = voi(inst, MU_SERVICE)
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
