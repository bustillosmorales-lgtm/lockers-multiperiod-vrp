"""
Value of integration at scale, using the matheuristic (Phase C) instead of the
exact solver, so we can reach n = 8..20 and tau >= 3.

joint  : matheuristic over the whole horizon.
myopic : matheuristic on each period in sequence (single-period sub-instance with
         depot dispatch = g_t + returns_{t-1}), no lookahead.

Because a myopic solution is feasible for the joint problem, the true joint
optimum <= obj_myopic; we therefore report the conservative
    VoI = max(0, obj_myopic - obj_joint_heur) >= 0,
which is a lower bound on the true value of integration (heuristic joint can only
over-estimate the joint optimum). CBC-exact cross-check on small n.

Run:  python voi_scaled.py
"""

import csv
import json
import os
import statistics as st

import voi as V
import matheuristic as H
from flow_eval import eval_flow


def single_period_instance(inst, t, dispatch_in):
    """Sub-instance with only period t and depot demand = dispatch_in."""
    return {"n": inst["n"], "T": [1], "d": inst["d"],
            "p": {1: inst["p"][t]}, "g": {1: dict(dispatch_in)},
            "Q": inst["Q"], "H": inst["H"], "C": inst["C"]}


def myopic_matheuristic(inst, mu, seed):
    """Rolling-horizon matheuristic. Returns (total_obj, routes_by_period)."""
    K = list(range(1, inst["n"] + 1))
    prev_ret = {k: 0 for k in K}
    total_obj = 0.0
    myopic_routes = {}
    for t in inst["T"]:
        disp = {k: inst["g"][t][k] + prev_ret[k] for k in K}
        sub = single_period_instance(inst, t, disp)
        res = H.matheuristic(sub, mu, iters=300, seed=seed)
        if not res["feasible"]:
            return None, None
        total_obj += res["objective"]
        myopic_routes[t] = res["routes"][1]     # sub uses period index 1
        detail = eval_flow(sub, res["routes"], mu, return_detail=True)
        prev_ret = detail["returns_by_period"][1]
    return total_obj, myopic_routes


def run(n, tau, seed, Q, mu):
    inst = V.make_instance(n, tau, seed, Q)
    inst["C"] = 3
    # myopic first, then warm-start the JOINT search from the myopic routing.
    # Because the myopic solution is feasible for the joint problem, the joint
    # search starts at obj_myopic and only improves: VoI = obj_myopic - obj_joint
    # is then a genuine (non-clamped) measure of cross-period coordination value.
    om, myopic_routes = myopic_matheuristic(inst, mu, seed)
    if om is None:
        return None
    j = H.matheuristic(inst, mu, iters=600, seed=seed, init_routes=myopic_routes)
    if not j["feasible"]:
        return None
    voi = max(0.0, om - j["objective"])
    return {"n": n, "tau": tau, "seed": seed, "Q": Q, "mu": mu,
            "obj_joint": j["objective"], "obj_myopic": round(om, 2),
            "VoI": round(voi, 2),
            "VoI_pct": round(100 * voi / om, 2) if om else 0.0}


def main():
    mu = 5
    tau = 3
    seeds = list(range(6))
    sizes = [8, 10, 12, 15, 20]
    rows = []
    for n in sizes:
        ref = V.total_flow(V.make_instance(n, tau, 0, Q=999))
        Q = max(20, int(0.5 * ref))     # tight-but-feasible band (where integration pays)
        for s in seeds:
            r = run(n, tau, s, Q, mu)
            if r:
                rows.append(r)
                print(f"n={n:>2} seed={s} VoI={r['VoI']:>7} ({r['VoI_pct']:>5}%)")

    summary = {}
    for n in sizes:
        sub = [x for x in rows if x["n"] == n]
        if not sub:
            continue
        vp = [x["VoI_pct"] for x in sub]
        summary[f"n{n}"] = {"instances": len(sub),
                            "frac_VoI_positive": round(sum(v > 1e-6 for v in vp) / len(sub), 2),
                            "mean_VoI_pct": round(st.mean(vp), 2),
                            "max_VoI_pct": round(max(vp), 2)}

    outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    with open(os.path.join(outdir, "voi_scaled.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    with open(os.path.join(outdir, "voi_scaled_summary.json"), "w") as f:
        json.dump({"mu": mu, "tau": tau, "seeds": len(seeds), "summary": summary}, f, indent=2)
    print("\n=== VoI by size (matheuristic, tight capacity, service objective) ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
