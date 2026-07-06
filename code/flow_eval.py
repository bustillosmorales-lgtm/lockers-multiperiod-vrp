"""
Exact flow evaluator for arbitrary per-period vehicle routes.

Given an instance and a routing (per period, a list of vehicle routes, each a
closed tour [0, ..lockers.., 0] partitioning all lockers), solve the
service-aware flow subproblem exactly:

    min  mu * R        s.t. constraints (5)-(11) + inter-period recirculation,

where R = total returned flow and the routing (hence distance) is fixed. This is
an LP in (y, z). Returns feasibility, R, distance, and objective dist + mu*R.

This is the inner exact evaluation used by the ALNS matheuristic: the outer
search proposes routings; this routine scores them exactly.
"""

import pulp


def route_arcs_by_period(routes_by_period):
    out = {}
    for t, routes in routes_by_period.items():
        arcs = set()
        for r in routes:
            for a, b in zip(r[:-1], r[1:]):
                arcs.add((a, b))
        out[t] = sorted(arcs)
    return out


def eval_flow(inst, routes_by_period, mu, return_detail=False):
    """Score a routing by its exact optimal service-aware flow.
    Returns dict: feasible, R, distance, objective (+ per-period returns if asked)."""
    n, T = inst["n"], inst["T"]
    K = list(range(1, n + 1))
    V = list(range(n + 1))
    d = inst["d"]
    arcs = route_arcs_by_period(routes_by_period)

    # distance is fixed by the routing
    distance = sum(d[(i, j)] for t in T for (i, j) in arcs[t])

    dout = {t: {i: [j for (ii, j) in arcs[t] if ii == i] for i in V} for t in T}
    din = {t: {i: [j for (j, ii) in arcs[t] if ii == i] for i in V} for t in T}

    prob = pulp.LpProblem("flow_eval", pulp.LpMinimize)
    z = {t: {(i, j): {k: pulp.LpVariable(f"z_{t}_{i}_{j}_{k}", lowBound=0)
                      for k in K}
             for (i, j) in arcs[t]} for t in T}
    y = {t: {k: pulp.LpVariable(f"y_{t}_{k}", lowBound=0) for k in K} for t in T}

    def ret(t, k):
        return pulp.lpSum(z[t][(i, 0)][k] for i in din[t][0])

    def disp(t, k):
        return pulp.lpSum(z[t][(0, j)][k] for j in dout[t][0])

    prob += mu * pulp.lpSum(ret(t, k) for t in T for k in K)  # min returns

    for t in T:
        for k in K:  # (5)
            prob += (pulp.lpSum(z[t][(i, k)][k] for i in din[t][k])
                     - pulp.lpSum(z[t][(k, j)][k] for j in dout[t][k]) == y[t][k])
        for j in K:  # (6)
            for k in K:
                if j == k:
                    continue
                prob += (pulp.lpSum(z[t][(j, i)][k] for i in dout[t][j])
                         - pulp.lpSum(z[t][(i, j)][k] for i in din[t][j])
                         == inst["p"][t][j][k])
        for (i, j) in arcs[t]:  # (7) capacity on used arcs
            prob += pulp.lpSum(z[t][(i, j)][k] for k in K) <= inst["Q"]
        for k in K:  # (8), (11)
            prob += y[t][k] <= inst["H"]
            prob += y[t][k] >= disp(t, k)

    for k in K:  # (9)
        prob += disp(T[0], k) == inst["g"][T[0]][k]
    for idx in range(1, len(T)):  # (10)
        t, tp = T[idx], T[idx - 1]
        for k in K:
            prob += disp(t, k) == inst["g"][t][k] + ret(tp, k)

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    feasible = pulp.LpStatus[prob.status] == "Optimal"
    if not feasible:
        return {"feasible": False, "R": None, "distance": distance,
                "objective": float("inf")}
    R = round(sum(ret(t, k).value() for t in T for k in K))
    res = {"feasible": True, "R": R, "distance": distance,
           "objective": round(distance + mu * R, 4)}
    if return_detail:
        res["returns_by_period"] = {t: {k: round(ret(t, k).value()) for k in K} for t in T}
        res["y_by_period"] = {t: {k: round(y[t][k].value()) for k in K} for t in T}
    return res
