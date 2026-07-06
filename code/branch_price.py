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


def _ng_neighbors(inst, ng_size):
    """Static ng-neighborhoods: the ng_size nearest lockers to each locker (by d).
    Symmetrized distance so one-way arcs do not distort proximity."""
    n = inst["n"]; K = list(range(1, n + 1)); d = inst["d"]
    N = {}
    for j in K:
        prox = sorted((k for k in K if k != j),
                      key=lambda k: d[(j, k)] + d[(k, j)])
        N[j] = frozenset(prox[:ng_size])
    return N


def _price_ng(inst, t, pi, om, sig, ng_size=7, N=None):
    """ng-route relaxation pricer (Baldacci-Mingozzi-Roberti) by forward labelling.

    A label is (last node j, ng-memory Pi subset of {j} u N_j): Pi holds the visited
    nodes that j 'remembers' and may not revisit; nodes outside N_j are forgotten, so
    the route may revisit them. This RELAXES elementarity, so the minimum reduced
    cost over ng-routes is <= that over elementary tours: the resulting CG bound is a
    VALID lower bound (weaker than the exact DP, but computable when 2^n is not).
    Route length is capped at n arcs, which keeps all elementary tours in scope and
    makes the labelling finite (no negative ng-cycle can be exploited without bound).

    State count per node <= 2^{ng_size+1}, so total work ~ O(n^2 * 2^{ng_size}) per
    length layer -- tractable well past the n<=15 ceiling of the bitmask DP.
    Returns (reduced_cost, order); order may be non-elementary (caller checks).
    """
    n = inst["n"]; K = list(range(1, n + 1)); d = inst["d"]; Q = inst["Q"]
    INF = float("inf")
    if N is None:
        N = _ng_neighbors(inst, ng_size)

    def dc(i, j):
        return d[(i, j)] + Q * sig[(t, i, j)]

    best = {}; par = {}
    for k in K:                                   # depot -> k
        st = (k, frozenset((k,)))
        best[st] = dc(0, k) - pi[(t, k)]; par[st] = (0, None)

    for _ in range(n - 1):                         # extend by one arc, length <= n
        updated = {}
        for (j, Pi), c in best.items():
            for m in K:
                if m == j or m in Pi:
                    continue
                nc = c + dc(j, m) - pi[(t, m)]
                st = (m, frozenset((m,)) | (Pi & N[m]))
                cur = updated.get(st)
                if nc < best.get(st, INF) and (cur is None or nc < cur[0]):
                    updated[st] = (nc, (j, Pi))
        if not updated:
            break
        changed = False
        for st, (nc, pp) in updated.items():
            if nc < best.get(st, INF):
                best[st] = nc; par[st] = pp; changed = True
        if not changed:
            break

    brc, bst = INF, None
    for (j, Pi), c in best.items():
        cc = c + dc(j, 0)
        if cc < brc:
            brc, bst = cc, (j, Pi)
    redcost = brc - om[t]

    order_rev, st = [], bst                         # reconstruct
    seen = 0
    while st is not None:
        j, _Pi = st
        order_rev.append(j)
        pp = par.get(st)
        seen += 1
        if pp is None or pp[0] == 0 or seen > n + 1:
            break
        st = (pp[0], pp[1])
    order = [0] + order_rev[::-1] + [0]
    return redcost, order


def _label_S(inst, t, pi, om, sig, S):
    """Forward labelling with memory restricted to a critical set S: a node in S may
    not be revisited; nodes outside S may be. State = (last node, visited cap S), so
    the state count is n * 2^{|S|}. Route length capped at n arcs. Returns
    (reduced_cost, order). This is the engine for DSSR below."""
    n = inst["n"]; K = list(range(1, n + 1)); d = inst["d"]; Q = inst["Q"]
    INF = float("inf")

    def dc(i, j):
        return d[(i, j)] + Q * sig[(t, i, j)]

    best = {}; par = {}
    for k in K:
        Pi = frozenset((k,)) if k in S else frozenset()
        st = (k, Pi)
        best[st] = dc(0, k) - pi[(t, k)]; par[st] = (0, None)
    for _ in range(n - 1):
        updated = {}
        for (j, Pi), c in best.items():
            for m in K:
                if m == j or m in Pi:
                    continue
                nc = c + dc(j, m) - pi[(t, m)]
                Pi2 = (Pi | frozenset((m,))) if m in S else Pi
                st = (m, Pi2); cur = updated.get(st)
                if nc < best.get(st, INF) and (cur is None or nc < cur[0]):
                    updated[st] = (nc, (j, Pi))
        if not updated:
            break
        changed = False
        for st, (nc, pp) in updated.items():
            if nc < best.get(st, INF):
                best[st] = nc; par[st] = pp; changed = True
        if not changed:
            break
    brc, bst = INF, None
    for (j, Pi), c in best.items():
        cc = c + dc(j, 0)
        if cc < brc:
            brc, bst = cc, (j, Pi)
    order_rev, st, seen = [], bst, 0
    while st is not None:
        j, _Pi = st
        order_rev.append(j)
        pp = par.get(st); seen += 1
        if pp is None or pp[0] == 0 or seen > n + 1:
            break
        st = (pp[0], pp[1])
    return brc - om[t], [0] + order_rev[::-1] + [0]


def _price_dssr(inst, t, pi, om, sig, max_S=18):
    """Decremental state-space relaxation: the EXACT elementary min-reduced-cost tour
    computed by repeated ng-style labelling. Start with an empty critical set S; solve;
    if the min route revisits a node, add the revisited nodes to S (forcing them
    elementary) and re-solve. On termination with an elementary route the reduced cost
    is EXACT, so the CG bound equals the bitmask DP -- but the state count is n*2^{|S|}
    with |S| the few nodes that actually cycle, so it runs past the n<=15 ceiling.
    If |S| exceeds max_S we stop and return the current (relaxed, still valid lower
    bound) reduced cost. Returns (reduced_cost, order, elementary, |S|)."""
    from collections import Counter
    S = set()
    while True:
        rc, order = _label_S(inst, t, pi, om, sig, frozenset(S))
        interior = order[1:-1]
        rep = [k for k, c in Counter(interior).items() if c > 1]
        if not rep:
            return rc, order, True, len(S)
        S.update(rep)
        if len(S) > max_S:
            return rc, order, False, len(S)


def cg_lower_bound(inst, mu, max_iter=40, verbose=False, return_cuts="3cycle",
                   pricer="milp", ng_size=7):
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
    N = _ng_neighbors(inst, ng_size) if pricer == "ng" else None
    best_lb = -float("inf")
    for it in range(max_iter):
        res = _solve_rmp(inst, mu, columns, rbound=rbound)
        if res is None:
            return None
        obj, pi, om, sig = res
        cbar_stars = []
        added = 0
        neg = False                       # a negative-reduced-cost route still exists
        for t in inst["T"]:
            if pricer == "dp":
                rc, order = _price_dp(inst, t, pi, om, sig)
            elif pricer == "ng":
                rc, order = _price_ng(inst, t, pi, om, sig, ng_size=ng_size, N=N)
            elif pricer == "dssr":
                rc, order, _elem, _sz = _price_dssr(inst, t, pi, om, sig)
            else:
                rc, order = _price(inst, t, pi, om, sig)
            cbar_stars.append(min(0.0, rc))
            if rc < -1e-6:
                neg = True
                interior = order[1:-1]
                # add the column only if the priced route is elementary; a
                # non-elementary ng-route still counts toward the Lagrangian bound
                if len(order) >= 3 and len(interior) == len(set(interior)):
                    columns[t].append(Route(inst, order))
                    added += 1
        lb = obj + inst["C"] * sum(cbar_stars)
        best_lb = max(best_lb, lb)
        if verbose:
            print(f"  it={it} RMP={obj:.1f} LB={lb:.1f} cols_added={added}")
        if not neg:
            return round(obj, 4)          # converged: no negative route -> RMP is the CG bound
        if added == 0:
            break                          # negative route exists but non-elementary: keep Lagrangian LB
    return round(best_lb, 4)
