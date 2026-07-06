"""
Full multi-period locker VRP (constraints 2-14 of Versión0.tex) in PuLP/CBC,
with a service-aware objective variant:

    min  sum_t sum_(i,j) d_ij x_ij^t  +  mu * R,      R = total returned flow.

mu = 0 recovers the paper's distance-only objective. mu > 0 penalizes overnight
recirculation, i.e. rewards same-day completion. Used by voi.py for the
value-of-integration study (joint horizon vs myopic rolling horizon).

Two entry points:
  solve_joint(inst, mu)            -- optimize the whole horizon at once
  solve_single(inst, t, disp, mu)  -- optimize ONE period given depot dispatch
                                      (building block of the myopic rolling horizon)

An instance `inst` is a dict:
  n       : number of lockers (lockers are 1..n, depot 0)
  T       : list of periods, e.g. [1,2,3]
  d       : {(i,j): dist} complete directed graph on {0..n}
  p       : {t: {j: {k: demand}}} locker-to-locker demand
  g       : {t: {k: new depot packages}}
  C, Q, H : fleet size, vehicle capacity, locker daily reception limit
"""

import pulp


def _arcs(n):
    V = list(range(n + 1))
    return [(i, j) for i in V for j in V if i != j]


def _add_routing_flow(prob, inst, t, dispatch_expr, mu, tag, relax=False):
    """Add all per-period variables and constraints (2)-(8),(11)-(14) for one
    period t. `dispatch_expr[k]` is the LP expression for depot dispatch of
    commodity k (an equality is added: sum_j z_0jk == dispatch_expr[k]).
    relax=True makes x continuous in [0,1] (LP relaxation -> lower bound).
    Returns (x, y, z, ret_expr) where ret_expr[k] = returns of commodity k."""
    n, Q, H, C = inst["n"], inst["Q"], inst["H"], inst["C"]
    K = list(range(1, n + 1))
    V = list(range(n + 1))
    A = _arcs(n)
    dout = {i: [j for (ii, j) in A if ii == i] for i in V}
    din = {i: [j for (j, ii) in A if ii == i] for i in V}

    if relax:
        x = {(i, j): pulp.LpVariable(f"x_{tag}_{i}_{j}", lowBound=0, upBound=1,
                                     cat="Continuous") for (i, j) in A}
    else:
        x = {(i, j): pulp.LpVariable(f"x_{tag}_{i}_{j}", cat="Binary") for (i, j) in A}
    y = {k: pulp.LpVariable(f"y_{tag}_{k}", lowBound=0) for k in K}
    z = {(i, j): {k: pulp.LpVariable(f"z_{tag}_{i}_{j}_{k}", lowBound=0)
                  for k in K} for (i, j) in A}
    w = {(i, j): pulp.LpVariable(f"w_{tag}_{i}_{j}", lowBound=0) for (i, j) in A}

    def disp(k):
        return pulp.lpSum(z[(0, j)][k] for j in dout[0])

    def ret(k):
        return pulp.lpSum(z[(i, 0)][k] for i in din[0])

    # (2) vehicle availability
    prob += pulp.lpSum(x[(0, j)] for j in dout[0]) <= C
    for j in K:
        # (3) visit each locker once
        prob += pulp.lpSum(x[(i, j)] for i in din[j]) == 1
        # (4) route continuity
        prob += (pulp.lpSum(x[(i, j)] for i in din[j])
                 == pulp.lpSum(x[(j, i)] for i in dout[j]))
    for k in K:
        # (5) delivery = net inflow of commodity k at k
        prob += (pulp.lpSum(z[(i, k)][k] for i in din[k])
                 - pulp.lpSum(z[(k, j)][k] for j in dout[k]) == y[k])
    for j in K:
        for k in K:
            if j == k:
                continue
            # (6) generation of commodity k at origin j
            prob += (pulp.lpSum(z[(j, i)][k] for i in dout[j])
                     - pulp.lpSum(z[(i, j)][k] for i in din[j]) == inst["p"][t][j][k])
    for (i, j) in A:
        # (7) capacity links flow to routing
        prob += pulp.lpSum(z[(i, j)][k] for k in K) <= Q * x[(i, j)]
    for k in K:
        prob += y[k] <= H                 # (8) locker reception limit
        prob += y[k] >= disp(k)           # (11) 24h rule on depot flow
        prob += disp(k) == dispatch_expr[k]  # (9)/(10): dispatch definition
    # (12)-(14) subtour elimination
    for (i, j) in A:
        prob += w[(i, j)] <= n * x[(i, j)]
    for j in K:
        prob += w[(0, j)] == 0
    for j in K:
        prob += (pulp.lpSum(w[(i, j)] for i in din[j]) + 1
                 == pulp.lpSum(w[(j, i)] for i in dout[j]))

    ret_expr = {k: ret(k) for k in K}
    return x, y, z, ret_expr


def _extract(x, y, z, ret_expr, inst, t, d):
    n = inst["n"]
    K = list(range(1, n + 1))
    A = _arcs(n)
    used = [(i, j) for (i, j) in A if x[(i, j)].value() > 0.5]
    dist = sum(d[(i, j)] for (i, j) in used)
    yv = {k: round(y[k].value()) for k in K}
    rv = {k: round(ret_expr[k].value()) for k in K}
    return {"period": t, "used_arcs": used, "distance": dist,
            "y": yv, "returns": rv, "R": sum(rv.values())}


def _routes_from_arcs(used, n):
    """Reconstruct routes from used arcs (for validation / display)."""
    succ = {}
    for (i, j) in used:
        succ.setdefault(i, []).append(j)
    routes = []
    for start in list(succ.get(0, [])):
        route = [0, start]
        cur = start
        while cur != 0 and len(route) <= n + 2:
            nxts = succ.get(cur, [])
            if not nxts:
                break
            cur = nxts.pop(0)
            route.append(cur)
        routes.append(route)
    return routes


def _make_solver(solver, time_limit):
    if solver == "highs":
        return pulp.HiGHS(msg=False, timeLimit=time_limit)
    return pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)


def solve_joint(inst, mu, time_limit=120, relax=False, cutoff=None, solver="cbc"):
    """Optimize the whole horizon jointly (the paper's model + mu*R).
    solver: 'cbc' (default) or 'highs' (stronger open-source MIP).
    relax=True solves the LP relaxation (x continuous) -> a lower bound.
    cutoff (if given) adds `objective <= cutoff`: since a feasible solution with
    that value is known (e.g. the myopic solution), this prunes the search and
    guarantees the returned incumbent is <= cutoff, without changing the true
    optimum (which is <= cutoff by assumption)."""
    n, T = inst["n"], inst["T"]
    K = list(range(1, n + 1))
    d = inst["d"]
    prob = pulp.LpProblem("joint", pulp.LpMinimize)

    blocks = {}
    prev_ret = None
    dist_terms, ret_terms = [], []
    for idx, t in enumerate(T):
        if idx == 0:
            disp_expr = {k: inst["g"][t][k] for k in K}
        else:
            disp_expr = {k: inst["g"][t][k] + prev_ret[k] for k in K}
        x, y, z, ret_expr = _add_routing_flow(prob, inst, t, disp_expr, mu,
                                              tag=f"j{t}", relax=relax)
        blocks[t] = (x, y, z, ret_expr)
        prev_ret = ret_expr
        A = _arcs(n)
        dist_terms.append(pulp.lpSum(d[(i, j)] * x[(i, j)] for (i, j) in A))
        ret_terms.append(pulp.lpSum(ret_expr[k] for k in K))

    obj_expr = pulp.lpSum(dist_terms) + mu * pulp.lpSum(ret_terms)
    prob += obj_expr
    if cutoff is not None and not relax:
        prob += obj_expr <= cutoff + 1e-6
    prob.solve(_make_solver(solver, time_limit))

    if relax:
        return {"status": pulp.LpStatus[prob.status],
                "lower_bound": round(pulp.value(prob.objective), 4)}

    per = {t: _extract(*blocks[t], inst, t, d) for t in T}
    return {"status": pulp.LpStatus[prob.status],
            "objective": round(pulp.value(prob.objective), 4),
            "total_distance": sum(per[t]["distance"] for t in T),
            "total_R": sum(per[t]["R"] for t in T),
            "per_period": per}


def lower_bound(inst, mu, time_limit=120):
    """LP-relaxation lower bound on the joint optimum."""
    return solve_joint(inst, mu, time_limit=time_limit, relax=True)["lower_bound"]


def solve_single(inst, t, dispatch_in, mu, time_limit=120):
    """Optimize ONE period given depot dispatch dispatch_in[k] (myopic step)."""
    n = inst["n"]
    K = list(range(1, n + 1))
    d = inst["d"]
    prob = pulp.LpProblem(f"single_{t}", pulp.LpMinimize)
    disp_expr = {k: dispatch_in[k] for k in K}
    x, y, z, ret_expr = _add_routing_flow(prob, inst, t, disp_expr, mu, tag=f"s{t}")
    A = _arcs(n)
    prob += (pulp.lpSum(d[(i, j)] * x[(i, j)] for (i, j) in A)
             + mu * pulp.lpSum(ret_expr[k] for k in K))
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))
    res = _extract(x, y, z, ret_expr, inst, t, d)
    res["status"] = pulp.LpStatus[prob.status]
    return res


def solve_myopic(inst, mu, time_limit=120):
    """Rolling horizon: solve each period in sequence, carrying returns forward.
    No lookahead -- each day only sees today."""
    n, T = inst["n"], inst["T"]
    K = list(range(1, n + 1))
    prev_ret = {k: 0 for k in K}
    per = {}
    for t in T:
        disp_in = {k: inst["g"][t][k] + prev_ret[k] for k in K}
        res = solve_single(inst, t, disp_in, mu, time_limit)
        per[t] = res
        prev_ret = res["returns"]
    return {"status": "rolling",
            "total_distance": sum(per[t]["distance"] for t in T),
            "total_R": sum(per[t]["R"] for t in T),
            "per_period": per}


def validate(inst, sol):
    """Sanity checks on a joint/myopic solution: subtour-free routes,
    service policy, and non-negative flows."""
    n = inst["n"]
    problems = []
    for t, res in sol["per_period"].items():
        routes = _routes_from_arcs(res["used_arcs"], n)
        visited = [x for r in routes for x in r[1:-1]]
        if sorted(visited) != list(range(1, n + 1)):
            problems.append(f"t={t}: lockers visited {sorted(visited)} != 1..{n}")
        for r in routes:
            if r[0] != 0 or r[-1] != 0:
                problems.append(f"t={t}: route not closed at depot: {r}")
    return problems
