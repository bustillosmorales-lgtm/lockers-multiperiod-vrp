"""
Column-generation (branch-and-price root) lower bound via a route-based
Dantzig-Wolfe reformulation of the multi-period locker VRP.

Master (restricted): pick route columns per period; each locker covered once,
at most C routes per period; commodity flow / returns live in the master and are
tied to the chosen routes by the capacity-linking constraint. Because route
columns are elementary tours by construction, the master LP is far stronger than
the arc-based relaxation -- this is the route to a tight, valid lower bound.

Master (period t):
  min  sum_{t,r} dist_r lam_{t,r} + mu * total returns
  s.t. sum_{r ni k} lam_{t,r} = 1                 (cover)      dual pi_{t,k}
       sum_r lam_{t,r} <= C                        (vehicles)  dual om_t   (<=0)
       sum_k z_{t,i,j,k} - Q sum_{r ni (i,j)} lam <= 0 (cap)   dual sig_{t,ij} (<=0)
       flow constraints (5),(6),(8),(9),(10),(11) in (z,y)

Pricing (period t): min-reduced-cost elementary tour, with
  reduced cost = sum_{(i,j) in r} (d_ij + Q*sig_{t,ij}) - sum_{k in r} pi_{t,k} - om_t.
Solved as a prize-collecting elementary tour MILP (MTZ subtours).

Valid Lagrangian bound each round:  LB = z_RMP + C * sum_t min(0, cbar*_t).
At convergence (no negative reduced cost) LB = z_RMP.

Deps: pulp (CBC). Slower than a labeling pricer, but correct and easy to verify.
"""

import pulp

from model_full import _arcs
from matheuristic import _nn_tour


class Route:
    def __init__(self, inst, order):
        self.order = order
        self.nodes = frozenset(order[1:-1])
        self.arcs = frozenset(zip(order[:-1], order[1:]))
        self.dist = sum(inst["d"][a] for a in self.arcs)


def _init_columns(inst, mu):
    """First feasible construction (fewest vehicles), as initial columns."""
    from flow_eval_fast import eval_flow_fast
    import random
    rng = random.Random(0)
    K = list(range(1, inst["n"] + 1))
    for veh in range(1, inst["C"] + 1):
        sol = {}
        for t in inst["T"]:
            groups = [[] for _ in range(veh)]
            order = sorted(K, key=lambda k: inst["d"][(0, k)])
            for idx, k in enumerate(order):
                groups[idx % veh].append(k)
            sol[t] = [_nn_tour(inst, g, rng) for g in groups if g]
        if eval_flow_fast(inst, sol, mu)["feasible"]:
            return {t: [Route(inst, r) for r in sol[t]] for t in inst["T"]}
    return None


def _solve_rmp(inst, mu, columns, rbound=None):
    n, T = inst["n"], inst["T"]
    K = list(range(1, n + 1))
    A = _arcs(n)
    din = {i: [j for (j, ii) in A if ii == i] for i in range(n + 1)}
    dout = {i: [j for (ii, j) in A if ii == i] for i in range(n + 1)}

    prob = pulp.LpProblem("rmp", pulp.LpMinimize)
    lam = {t: [pulp.LpVariable(f"lam_{t}_{c}", lowBound=0) for c in range(len(columns[t]))]
           for t in T}
    z = {t: {(i, j): {k: pulp.LpVariable(f"z_{t}_{i}_{j}_{k}", lowBound=0) for k in K}
             for (i, j) in A} for t in T}
    yv = {t: {k: pulp.LpVariable(f"y_{t}_{k}", lowBound=0) for k in K} for t in T}

    def ret(t, k):
        return pulp.lpSum(z[t][(i, 0)][k] for i in din[0])

    def disp(t, k):
        return pulp.lpSum(z[t][(0, j)][k] for j in dout[0])

    prob += (pulp.lpSum(columns[t][c].dist * lam[t][c] for t in T for c in range(len(columns[t])))
             + mu * pulp.lpSum(ret(t, k) for t in T for k in K))

    cover, veh, cap = {}, {}, {}
    for t in T:
        for k in K:
            con = pulp.lpSum(lam[t][c] for c in range(len(columns[t]))
                             if k in columns[t][c].nodes) == 1
            prob += con, f"cover_{t}_{k}"; cover[(t, k)] = con
        con = pulp.lpSum(lam[t]) <= inst["C"]
        prob += con, f"veh_{t}"; veh[t] = con
        for (i, j) in A:
            con = (pulp.lpSum(z[t][(i, j)][k] for k in K)
                   - inst["Q"] * pulp.lpSum(lam[t][c] for c in range(len(columns[t]))
                                            if (i, j) in columns[t][c].arcs) <= 0)
            prob += con, f"cap_{t}_{i}_{j}"; cap[(t, i, j)] = con
        for k in K:  # (5)
            prob += (pulp.lpSum(z[t][(i, k)][k] for i in din[k])
                     - pulp.lpSum(z[t][(k, j)][k] for j in dout[k]) == yv[t][k])
        for j in K:  # (6)
            for k in K:
                if j == k:
                    continue
                prob += (pulp.lpSum(z[t][(j, i)][k] for i in dout[j])
                         - pulp.lpSum(z[t][(i, j)][k] for i in din[j]) == inst["p"][t][j][k])
        for k in K:  # (8),(11)
            prob += yv[t][k] <= inst["H"]
            prob += yv[t][k] >= disp(t, k)
    for k in K:  # (9)
        prob += disp(T[0], k) == inst["g"][T[0]][k]
    for idx in range(1, len(T)):  # (10)
        t, tp = T[idx], T[idx - 1]
        for k in K:
            prob += disp(t, k) == inst["g"][t][k] + ret(tp, k)

    if rbound is not None:  # valid inequalities on the return flow (per period)
        for t in T:
            prob += pulp.lpSum(ret(t, k) for k in K) >= rbound[t]

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        return None
    obj = pulp.value(prob.objective)
    pi = {(t, k): (cover[(t, k)].pi or 0.0) for t in T for k in K}
    om = {t: (veh[t].pi or 0.0) for t in T}
    sig = {(t, i, j): (cap[(t, i, j)].pi or 0.0) for t in T for (i, j) in A}
    return obj, pi, om, sig


def _price(inst, t, pi, om, sig, tl=20):
    """Min-reduced-cost elementary tour in period t. Returns (redcost, order)."""
    n = inst["n"]
    K = list(range(1, n + 1))
    A = _arcs(n)
    d = inst["d"]
    din = {i: [j for (j, ii) in A if ii == i] for i in range(n + 1)}
    dout = {i: [j for (ii, j) in A if ii == i] for i in range(n + 1)}

    prob = pulp.LpProblem("price", pulp.LpMinimize)
    a = {(i, j): pulp.LpVariable(f"a_{i}_{j}", cat="Binary") for (i, j) in A}
    u = {k: pulp.LpVariable(f"u_{k}", cat="Binary") for k in K}
    o = {k: pulp.LpVariable(f"o_{k}", lowBound=1, upBound=n) for k in K}

    prob += pulp.lpSum(a[(0, j)] for j in K) == 1
    prob += pulp.lpSum(a[(i, 0)] for i in K) == 1
    for k in K:
        prob += pulp.lpSum(a[(i, k)] for i in din[k]) == u[k]
        prob += pulp.lpSum(a[(k, j)] for j in dout[k]) == u[k]
    prob += pulp.lpSum(u[k] for k in K) >= 1
    for (i, j) in A:
        if i != 0 and j != 0:
            prob += o[i] - o[j] + n * a[(i, j)] <= n - 1

    dprime = {(i, j): d[(i, j)] + inst["Q"] * sig[(t, i, j)] for (i, j) in A}
    prob += (pulp.lpSum(a[(i, j)] * dprime[(i, j)] for (i, j) in A)
             - pulp.lpSum(u[k] * pi[(t, k)] for k in K))
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=tl))
    redcost = pulp.value(prob.objective) - om[t]

    # reconstruct order
    succ = {i: j for (i, j) in A if a[(i, j)].value() and a[(i, j)].value() > 0.5}
    order = [0]
    cur = 0
    while True:
        nxt = succ.get(cur)
        if nxt is None or nxt == 0:
            order.append(0)
            break
        order.append(nxt)
        cur = nxt
        if len(order) > n + 2:
            break
    return redcost, order


def _price_dp(inst, t, pi, om, sig):
    """Exact pricer by bitmask DP (Held-Karp) for the elementary min-reduced-cost
    tour. O(2^n n^2) -- replaces the MILP pricer and scales to ~n<=18, covering
    more of the matheuristic's range. Returns (reduced_cost, order)."""
    n = inst["n"]
    K = range(1, n + 1)
    d = inst["d"]
    Q = inst["Q"]
    INF = float("inf")

    def dp_cost(i, j):
        return d[(i, j)] + Q * sig[(t, i, j)]

    size = 1 << n
    dp = [[INF] * (n + 1) for _ in range(size)]
    par = [[-1] * (n + 1) for _ in range(size)]
    for k in K:                                   # 0 -> k
        m = 1 << (k - 1)
        dp[m][k] = dp_cost(0, k) - pi[(t, k)]
        par[m][k] = 0
    for mask in range(size):
        row = dp[mask]
        for j in K:
            base = row[j]
            if base == INF or not (mask >> (j - 1)) & 1:
                continue
            for m2 in K:
                if (mask >> (m2 - 1)) & 1:
                    continue
                nm = mask | (1 << (m2 - 1))
                c = base + dp_cost(j, m2) - pi[(t, m2)]
                if c < dp[nm][m2]:
                    dp[nm][m2] = c
                    par[nm][m2] = j
    best, bmask, bj = INF, None, None
    for mask in range(size):
        row = dp[mask]
        for j in K:
            if row[j] == INF:
                continue
            c = row[j] + dp_cost(j, 0)             # close tour j -> 0
            if c < best:
                best, bmask, bj = c, mask, j
    redcost = best - om[t]
    rev, mask, j = [], bmask, bj                   # reconstruct
    while j != 0 and j != -1:
        rev.append(j)
        pj = par[mask][j]
        mask &= ~(1 << (j - 1))
        j = pj
    order = [0] + rev[::-1] + [0]
    return redcost, order


def cg_lower_bound(inst, mu, max_iter=40, verbose=False, return_cuts="3cycle",
                   pricer="milp"):
    """return_cuts: None (off), '2cycle', or '3cycle' return valid inequalities."""
    columns = _init_columns(inst, mu)
    if columns is None:
        return None
    rbound = None
    if return_cuts == "2cycle":
        from valid_ineq import bidir_return_bound
        rbound = bidir_return_bound(inst)
    elif return_cuts == "3cycle":
        from valid_ineq import return_bound_3cycle
        rbound = return_bound_3cycle(inst)
    best_lb = -float("inf")
    for it in range(max_iter):
        res = _solve_rmp(inst, mu, columns, rbound=rbound)
        if res is None:
            return None
        obj, pi, om, sig = res
        cbar_stars = []
        added = 0
        for t in inst["T"]:
            rc, order = (_price_dp(inst, t, pi, om, sig) if pricer == "dp"
                         else _price(inst, t, pi, om, sig))
            cbar_stars.append(min(0.0, rc))
            if rc < -1e-6 and len(order) >= 3:
                columns[t].append(Route(inst, order))
                added += 1
        lb = obj + inst["C"] * sum(cbar_stars)
        best_lb = max(best_lb, lb)
        if verbose:
            print(f"  it={it} RMP={obj:.1f} LB={lb:.1f} cols_added={added}")
        if added == 0:
            return round(obj, 4)          # converged: RMP obj is the CG bound
    return round(best_lb, 4)
