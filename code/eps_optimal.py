"""
Certify (rather than assert) that the identified interval of a reported indicator
widens monotonically over the EPS-optimal set -- so non-identification is not a
knife-edge phenomenon confined to exact ties. On a small instance with known
distances we compute the range of same-day deliveries Y over {routings+flows with
total distance <= (1+eps) z*} for a sweep of eps; the interval can only grow with eps.

(The paper's own instances do not publish distances, so this uses a synthetic
instance, exactly as optface_rho.py does.)

Output: results/eps_optimal.json
"""
import json
import os

import pulp

import voi as V
import model_full as M

TAU = 2
N = 5
SEED = 11            # the instance with the widest exact-optimal identified set in optface_rho
EPS = [0.0, 0.01, 0.02, 0.05, 0.10]


def y_extreme(inst, dcap, maximize):
    """min/max total same-day deliveries Y over {total distance <= dcap}."""
    n, T = inst["n"], inst["T"]
    K = list(range(1, n + 1))
    d = inst["d"]; A = M._arcs(n)
    sense = pulp.LpMaximize if maximize else pulp.LpMinimize
    prob = pulp.LpProblem("yeps", sense)
    prev_ret = None; dist_terms = []; y_terms = []
    for idx, t in enumerate(T):
        disp = {k: inst["g"][t][k] for k in K} if idx == 0 else \
               {k: inst["g"][t][k] + prev_ret[k] for k in K}
        x, y, z, ret_expr = M._add_routing_flow(prob, inst, t, disp, 0, tag=f"e{t}")
        prev_ret = ret_expr
        dist_terms.append(pulp.lpSum(d[(i, j)] * x[(i, j)] for (i, j) in A))
        y_terms.append(pulp.lpSum(y[k] for k in K))
    prob += pulp.lpSum(y_terms)
    prob += pulp.lpSum(dist_terms) <= dcap
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        return None
    return round(pulp.value(pulp.lpSum(y_terms)), 2)


def main():
    ref = V.total_flow(V.make_instance(N, TAU, SEED, Q=999))
    Q = max(30, int(0.6 * ref))
    inst = V.make_instance(N, TAU, SEED, Q=Q); inst["C"] = 2
    j = M.solve_joint(inst, 0, time_limit=120)
    zstar = j["total_distance"]
    rows = []
    prev_w = -1
    for eps in EPS:
        dcap = zstar * (1 + eps)
        lo = y_extreme(inst, dcap, False)
        hi = y_extreme(inst, dcap, True)
        w = round(hi - lo, 2)
        rows.append({"eps": eps, "dist_cap": round(dcap, 2),
                     "Y_interval": [lo, hi], "width": w,
                     "monotone_nondecreasing": w >= prev_w - 1e-9})
        prev_w = w
        print(f"eps={eps:.2f}: distance<= {dcap:.1f}  Y in [{lo},{hi}]  width {w}", flush=True)
    out = {"n": N, "tau": TAU, "seed": SEED, "z_star": zstar,
           "indicator": "same-day deliveries Y",
           "widens_monotonically": all(r["monotone_nondecreasing"] for r in rows),
           "rows": rows}
    outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    with open(os.path.join(outdir, "eps_optimal.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwidens monotonically with eps:", out["widens_monotonically"])


if __name__ == "__main__":
    main()
