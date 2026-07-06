"""Confirmation run: does the value of integration strengthen with the horizon?
Tight capacity, service objective (mu=5), tau=3, a few seeds. Compare to tau=2.
"""
import json, os, statistics as st
import model_full as M
import voi as V

def run(tau, seeds, Q, mu=5):
    out = []
    for s in seeds:
        inst = V.make_instance(n=5, tau=tau, seed=s, Q=Q)
        j = M.solve_joint(inst, mu, time_limit=90)
        if j["status"] == "Infeasible":
            continue
        m = M.solve_myopic(inst, mu, time_limit=90)
        oj, om = V.obj_of(j, mu), V.obj_of(m, mu)
        out.append({"seed": s, "VoI": round(om - oj, 2),
                    "VoI_pct": round(100*(om-oj)/oj, 2) if oj else 0,
                    "R_joint": j["total_R"], "R_myopic": m["total_R"],
                    "joint_status": j["status"]})
    return out

if __name__ == "__main__":
    seeds = list(range(8))
    Q = 22
    res = {"tau2": run(2, seeds, Q), "tau3": run(3, seeds, Q)}
    for k, rows in res.items():
        pos = sum(r["VoI"] > 1e-6 for r in rows)
        mx = max((r["VoI_pct"] for r in rows), default=0)
        mean = round(st.mean([r["VoI_pct"] for r in rows]), 2) if rows else 0
        print(f"{k}: n={len(rows)} VoI>0 in {pos}/{len(rows)} "
              f"mean={mean}% max={mx}%")
    outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    with open(os.path.join(outdir, "voi_tau_comparison.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("written voi_tau_comparison.json")
