"""
Certificate: non-identifiability of the return-flow ratio rho in the
distance-only multi-period locker VRP of De Santis (Versión0.tex).

The objective (1)  min sum_t sum_(i,j) d_ij x_ij^t  depends ONLY on the routing
variables x. Delivery (y), commodity flow (z) and load (w) never enter it, and
constraint (11) forces same-day delivery ONLY for depot-dispatched flow
(y_k >= sum_j z_0jk). A locker-origin package p_jk whose destination k is
visited AFTER j on the route may be delivered same-day OR carried back to the
depot at identical cost.

So: fix the routing to the paper's OWN published optimal routes and the flow
subproblem alone admits an INTERVAL of same-day-delivery totals -- hence an
interval of rho -- all optimal. This certifies rho is a tie-breaking artifact,
not a model output. No distance matrix is needed (routing is fixed to the
published routes), so the certificate is fully reproducible from the manuscript.

Two instances, both transcribed verbatim from Versión0.tex:
  - small  : Section 5, 5 lockers, tau=2
  - kandoo : Section 7 case study, 15 lockers, tau=2

Run:  python certify_rho.py
Deps: pulp (bundled CBC).
"""

import json
import os
import pulp


def route_arcs(routes):
    arcs = set()
    for r in routes:
        for a, b in zip(r[:-1], r[1:]):
            arcs.add((a, b))
    return sorted(arcs)


def solve_extreme(inst, sense):
    """Min or max total returned flow R over the flow polytope, routing fixed."""
    K, T, p, g, Q, H, routes = (inst["K"], inst["T"], inst["p"], inst["g"],
                                inst["Q"], inst["H"], inst["routes"])
    V = [0] + K

    arcs = {t: route_arcs(routes[t]) for t in T}
    dout = {t: {i: [j for (ii, j) in arcs[t] if ii == i] for i in V} for t in T}
    din = {t: {i: [j for (j, ii) in arcs[t] if ii == i] for i in V} for t in T}

    prob = pulp.LpProblem("rho_certificate", sense)
    z = {t: {(i, j): {k: pulp.LpVariable(f"z_{t}_{i}_{j}_{k}", lowBound=0)
                      for k in K}
             for (i, j) in arcs[t]} for t in T}
    y = {t: {k: pulp.LpVariable(f"y_{t}_{k}", lowBound=0) for k in K} for t in T}

    def ret(t, k):   # returned flow to depot for commodity k in period t
        return pulp.lpSum(z[t][(i, 0)][k] for i in din[t][0])

    def disp(t, k):  # dispatched flow from depot for commodity k in period t
        return pulp.lpSum(z[t][(0, j)][k] for j in dout[t][0])

    prob += pulp.lpSum(ret(t, k) for t in T for k in K)  # objective: min/max R

    for t in T:
        for k in K:  # (5) delivery = net inflow of commodity k at node k
            inflow = pulp.lpSum(z[t][(i, k)][k] for i in din[t][k])
            outflow = pulp.lpSum(z[t][(k, j)][k] for j in dout[t][k])
            prob += (inflow - outflow == y[t][k])
        for j in K:  # (6) generation of commodity k at origin locker j (j != k)
            for k in K:
                if j == k:
                    continue
                out_jk = pulp.lpSum(z[t][(j, i)][k] for i in dout[t][j])
                in_jk = pulp.lpSum(z[t][(i, j)][k] for i in din[t][j])
                prob += (out_jk - in_jk == p[t][j][k])
        for (i, j) in arcs[t]:  # (7) vehicle capacity on each used arc (x=1)
            prob += (pulp.lpSum(z[t][(i, j)][k] for k in K) <= Q)
        for k in K:  # (8) locker daily reception limit
            prob += (y[t][k] <= H)
        for k in K:  # (11) all depot-dispatched flow delivered same day
            prob += (y[t][k] >= disp(t, k))

    for k in K:  # (9) first-period depot dispatch = g^1
        prob += (disp(T[0], k) == g[T[0]][k])
    for idx in range(1, len(T)):  # (10) depot recirculation
        t, tp = T[idx], T[idx - 1]
        for k in K:
            prob += (disp(t, k) == g[t][k] + ret(tp, k))

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    Yval = sum(y[t][k].value() for t in T for k in K)
    Rval = sum(ret(t, k).value() for t in T for k in K)
    rho = Rval / (Yval + Rval) if (Yval + Rval) > 0 else float("nan")
    y_by_period = {t: [round(y[t][k].value()) for k in K] for t in T}
    return {"status": pulp.LpStatus[prob.status],
            "total_delivered_Y": round(Yval),
            "total_returned_R": round(Rval),
            "rho": round(rho, 6),
            "y_by_period": y_by_period}


def certify(inst):
    lo = solve_extreme(inst, pulp.LpMinimize)   # min returns -> max delivery
    hi = solve_extreme(inst, pulp.LpMaximize)   # max returns -> min delivery
    Y_max, Y_min = lo["total_delivered_Y"], hi["total_delivered_Y"]
    rho_min, rho_max = lo["rho"], hi["rho"]
    rep = inst["paper_reported_total_delivery"]
    where = ("Y_min (paper reports the LEAST same-day delivery / MOST recirculation)"
             if abs(Y_min - rep) < 1e-6 else
             "Y_max (paper reports the MOST same-day delivery / LEAST recirculation)"
             if abs(Y_max - rep) < 1e-6 else "interior")
    return {
        "instance": inst["name"],
        "lockers": len(inst["K"]),
        "routing": "fixed to the paper's published optimal routes",
        "delivery_interval_[Ymin,Ymax]": [Y_min, Y_max],
        "rho_identified_set_[rho_min,rho_max]": [rho_min, rho_max],
        "rho_width": round(rho_max - rho_min, 6),
        "paper_reported_total_delivery": rep,
        "paper_sits_at": where,
        "extreme_min_returns": lo,
        "extreme_max_returns": hi,
    }


# ---------------------------------------------------------------------------
# Instance data (verbatim from Versión0.tex)
# ---------------------------------------------------------------------------
def _mat(rows):
    """rows: list of 1..n lists; return {j:{k:val}} with 1-based indices."""
    return {j + 1: {k + 1: rows[j][k] for k in range(len(rows[j]))}
            for j in range(len(rows))}


SMALL = {
    "name": "small_5lockers_tau2",
    "K": [1, 2, 3, 4, 5],
    "T": [1, 2],
    "p": {
        1: _mat([[0, 7, 9, 5, 4], [7, 0, 4, 7, 7], [5, 8, 0, 4, 8],
                 [5, 2, 6, 0, 12], [10, 7, 3, 5, 0]]),
        2: _mat([[0, 8, 7, 6, 4], [6, 0, 5, 9, 5], [4, 7, 0, 5, 9],
                 [6, 3, 5, 0, 11], [9, 6, 4, 6, 0]]),
    },
    "g": {1: {1: 10, 2: 5, 3: 7, 4: 4, 5: 2},
          2: {1: 6, 2: 3, 3: 4, 4: 5, 5: 1}},
    "Q": 60, "H": 40,
    "routes": {1: [[0, 3, 1, 0], [0, 5, 2, 4, 0]],
               2: [[0, 3, 1, 0], [0, 5, 2, 4, 0]]},
    "paper_reported_total_delivery": 181,   # y1 sum 52 + y2 sum 129
}

KANDOO = {
    "name": "kandoo_15lockers_tau2",
    "K": list(range(1, 16)),
    "T": [1, 2],
    "p": {
        1: _mat([
            [0, 1, 1, 1, 1, 1, 0, 2, 1, 3, 0, 1, 3, 0, 0],
            [0, 0, 3, 1, 3, 0, 0, 3, 2, 0, 2, 0, 2, 0, 2],
            [0, 0, 0, 2, 2, 1, 1, 2, 0, 1, 2, 0, 1, 0, 0],
            [2, 1, 1, 0, 2, 2, 2, 3, 2, 1, 1, 2, 1, 1, 1],
            [0, 1, 1, 2, 0, 1, 2, 0, 2, 3, 2, 0, 0, 0, 0],
            [0, 2, 3, 0, 2, 0, 1, 0, 2, 0, 2, 1, 1, 1, 2],
            [2, 1, 2, 1, 1, 0, 0, 0, 2, 0, 3, 2, 1, 2, 0],
            [2, 1, 0, 0, 2, 2, 2, 0, 2, 0, 0, 0, 2, 1, 2],
            [2, 1, 0, 1, 0, 0, 0, 0, 0, 2, 1, 2, 1, 0, 2],
            [2, 3, 0, 1, 2, 1, 2, 1, 2, 0, 0, 1, 3, 1, 0],
            [3, 1, 0, 0, 1, 0, 1, 2, 1, 2, 0, 0, 2, 0, 1],
            [2, 2, 0, 1, 1, 1, 1, 0, 2, 0, 0, 0, 1, 1, 0],
            [3, 2, 2, 0, 3, 0, 1, 1, 3, 0, 2, 0, 0, 1, 2],
            [1, 2, 1, 2, 1, 0, 3, 0, 2, 1, 2, 1, 0, 0, 3],
            [2, 2, 0, 0, 0, 0, 0, 2, 2, 1, 1, 2, 2, 1, 0],
        ]),
        2: _mat([
            [0, 1, 1, 1, 1, 1, 0, 2, 1, 3, 0, 1, 3, 0, 0],
            [0, 0, 3, 1, 2, 0, 0, 3, 1, 0, 2, 0, 2, 0, 2],
            [0, 0, 0, 2, 2, 1, 1, 2, 0, 1, 2, 0, 1, 0, 0],
            [2, 1, 1, 0, 2, 2, 2, 3, 1, 1, 1, 2, 1, 1, 1],
            [0, 1, 1, 2, 0, 1, 2, 0, 2, 3, 2, 0, 0, 0, 0],
            [0, 2, 3, 0, 2, 0, 1, 0, 1, 0, 2, 1, 1, 1, 2],
            [2, 1, 2, 1, 1, 0, 0, 0, 2, 0, 3, 2, 1, 2, 0],
            [2, 1, 0, 0, 2, 2, 2, 0, 1, 0, 0, 0, 2, 1, 2],
            [2, 1, 0, 1, 0, 0, 0, 0, 0, 2, 1, 2, 1, 0, 2],
            [2, 3, 0, 1, 2, 1, 2, 1, 1, 0, 0, 1, 3, 1, 0],
            [2, 1, 0, 0, 1, 0, 1, 2, 1, 2, 0, 0, 2, 0, 1],
            [2, 2, 0, 1, 1, 1, 1, 0, 2, 0, 0, 0, 1, 1, 0],
            [3, 2, 2, 0, 3, 0, 1, 1, 3, 0, 2, 0, 0, 1, 2],
            [1, 2, 1, 2, 1, 0, 3, 0, 2, 1, 2, 1, 0, 0, 3],
            [2, 2, 0, 0, 0, 0, 0, 2, 2, 1, 1, 2, 2, 1, 0],
        ]),
    },
    "g": {1: {1: 6, 2: 8, 3: 5, 4: 4, 5: 7, 6: 3, 7: 5, 8: 6, 9: 4, 10: 5,
              11: 3, 12: 4, 13: 6, 14: 2, 15: 5},
          2: {k: 0 for k in range(1, 16)}},
    "Q": 125, "H": 30,
    "routes": {
        1: [[0, 5, 4, 6, 2, 3, 11, 10, 8, 9, 15, 13, 14, 1, 12, 7, 0]],
        2: [[0, 5, 6, 4, 2, 3, 10, 11, 9, 8, 15, 13, 14, 1, 12, 7, 0]],
    },
    "paper_reported_total_delivery": 435,   # day1 204 + day2 231
}


def main():
    results = [certify(SMALL), certify(KANDOO)]
    print(json.dumps(results, indent=2))
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "results", "rho_certificate.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWritten: {out}")


if __name__ == "__main__":
    main()
