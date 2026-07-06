"""
Instantiate Proposition prop:optface: compute the return-flow ratio's identified set
over the FULL mixed-integer optimal face (all cost-optimal routings), and compare it
to the fixed-single-routing inner bound (the two-LP interval the certificate reports).

For a small instance with known distances:
  1. z* = minimum total routing distance (distance-only objective, proven at n=5).
  2. inner bound: fix ONE optimal routing, min/max rho over its flow polytope
     (Dinkelbach at fixed routing) -> the classical two-LP interval.
  3. full optimal face: min/max rho over { (x,f) : total_dist(x) = z*, all constraints }
     -- a mixed-integer program (Dinkelbach outer loop, MILP inner) that ranges over
     EVERY cost-optimal routing, not one.
If the full interval is strictly wider than the inner bound, the hard object of
prop:optface is real (there are multiple optimal routings whose flow polytopes give
different rho ranges). If it collapses, the optimal routing is effectively unique and
the inner bound is exact -- also worth reporting.

Output: results/optface_rho.json
"""
import json
import os

import pulp

import voi as V
import model_full as M
import certify_rho as CR

TAU = 2
EPS = 1e-6


def optimal_distance(inst):
    j = M.solve_joint(inst, 0, time_limit=300, solver="cbc")
    return j["total_distance"], j["status"], j


def inner_bound(inst, jsol):
    """Fixed-routing two-LP (Dinkelbach) rho interval at one optimal routing."""
    routes = {t: [[i, j] for (i, j) in jsol["per_period"][t]["used_arcs"]]
              for t in inst["T"]}
    ci = dict(inst); ci["K"] = list(range(1, inst["n"] + 1)); ci["routes"] = routes
    lo = CR.solve_rho_extreme(ci, maximize=False)
    hi = CR.solve_rho_extreme(ci, maximize=True)
    return lo, hi


def rho_over_optface(inst, zstar, maximize, iters=50, tol=1e-9):
    """Exact min/max of rho over the full optimal face {total_dist == z*} by
    Dinkelbach; each inner problem is a MILP over routing + flow."""
    n, T = inst["n"], inst["T"]
    K = list(range(1, n + 1))
    d = inst["d"]
    A = M._arcs(n)
    lam = 0.0
    Rv = Yv = 0.0
    for _ in range(iters):
        sense = pulp.LpMaximize if maximize else pulp.LpMinimize
        prob = pulp.LpProblem("optface", sense)
        prev_ret = None
        dist_terms, y_terms, ret_terms = [], [], []
        for idx, t in enumerate(T):
            if idx == 0:
                disp = {k: inst["g"][t][k] for k in K}
            else:
                disp = {k: inst["g"][t][k] + prev_ret[k] for k in K}
            x, y, z, ret_expr = M._add_routing_flow(prob, inst, t, disp, 0, tag=f"of{t}")
            prev_ret = ret_expr
            dist_terms.append(pulp.lpSum(d[(i, j)] * x[(i, j)] for (i, j) in A))
            y_terms.append(pulp.lpSum(y[k] for k in K))
            ret_terms.append(pulp.lpSum(ret_expr[k] for k in K))
        R = pulp.lpSum(ret_terms)
        Ytot = pulp.lpSum(y_terms)
        prob += R - lam * (Ytot + R)                       # Dinkelbach objective
        prob += pulp.lpSum(dist_terms) == zstar            # stay on the optimal face
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        if pulp.LpStatus[prob.status] != "Optimal":
            return None
        Rv = pulp.value(R)
        Yv = pulp.value(Ytot)
        new_lam = Rv / (Yv + Rv) if (Yv + Rv) > 0 else 0.0
        if abs(new_lam - lam) < tol:
            lam = new_lam
            break
        lam = new_lam
    return round(lam, 6)


def run_instance(n, seed, Q):
    inst = V.make_instance(n, TAU, seed, Q=Q); inst["C"] = 2
    zstar, status, jsol = optimal_distance(inst)
    if status != "Optimal":
        return None
    lo_in, hi_in = inner_bound(inst, jsol)
    lo_full = rho_over_optface(inst, zstar, maximize=False)
    hi_full = rho_over_optface(inst, zstar, maximize=True)
    if lo_full is None or hi_full is None:
        return None
    return {
        "n": n, "seed": seed, "Q": Q, "z_star": zstar,
        "inner_bound_[lo,hi]": [lo_in, hi_in],
        "inner_width": round(hi_in - lo_in, 6),
        "optface_[lo,hi]": [lo_full, hi_full],
        "optface_width": round(hi_full - lo_full, 6),
        "optface_strictly_wider": (lo_full < lo_in - 1e-4) or (hi_full > hi_in + 1e-4),
    }


def main():
    rows = []
    # scan small instances (integer-rounded distances make routing ties, hence |X*|>1, common)
    for seed in range(12):
        r = run_instance(5, seed, Q=40)
        if r:
            rows.append(r)
            print(f"seed {seed}: inner {r['inner_bound_[lo,hi]']} (w={r['inner_width']})  "
                  f"optface {r['optface_[lo,hi]']} (w={r['optface_width']})  "
                  f"wider={r['optface_strictly_wider']}", flush=True)
    wider = [r for r in rows if r["optface_strictly_wider"]]
    summary = {
        "n_instances": len(rows),
        "n_optface_wider_than_inner": len(wider),
        "max_inner_width": round(max((r["inner_width"] for r in rows), default=0), 4),
        "max_optface_width": round(max((r["optface_width"] for r in rows), default=0), 4),
        "example_wider": wider[0] if wider else None,
    }
    outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    with open(os.path.join(outdir, "optface_rho.json"), "w") as f:
        json.dump({"tau": TAU, "summary": summary, "rows": rows}, f, indent=2)
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
