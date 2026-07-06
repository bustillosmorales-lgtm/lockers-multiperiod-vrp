"""
Valid inequalities on the return flow, to tighten the service-penalty part of the
lower bound (the residual gap the branch-and-price leaves untouched).

Bidirectional-demand return bound (routing-independent).
For any two lockers a, b, the packages that pair generates and that CANNOT be
completed same-day are at least min(p_ab, p_ba), whatever the routing:
  - same route: exactly one of a,b is visited first, so the backward-direction
    demand (>= min) returns to the depot;
  - different routes: neither reaches the other, so BOTH directions return
    (>= min as well).
Returns in period t come only from period-t locker pickups (depot-dispatched flow
must be delivered same-day, constraint (11)), so the bound applies per period:

    sum_k R_{t,k}  >=  B_t := sum_{a<b} min(p_ab^t, p_ba^t).

This is the 2-cycle lower bound of the underlying linear-ordering / minimum
feedback-arc-set structure; higher-order (3-cycle) inequalities would tighten it
further.
"""


def bidir_return_bound(inst):
    """2-cycle bound {t: B_t}: sum_{a<b} min(p_ab, p_ba). Routing-independent
    lower bound on per-period returns (feedback-arc-set LP with only 2-cycles)."""
    K = range(1, inst["n"] + 1)
    out = {}
    for t in inst["T"]:
        p = inst["p"][t]
        out[t] = sum(min(p[a][b], p[b][a])
                     for a in K for b in K if a < b)
    return out


def return_bound_3cycle(inst):
    """3-cycle bound {t: B_t}: LP value of the minimum-feedback-arc-set relaxation
    with 2-cycle equalities AND 3-cycle inequalities.

    Returns = backward-direction pickups over the visiting order. Over any single
    order (the minimum-return configuration; splits only add returns), the min
    total backward weight is the linear-ordering / MFAS optimum. Its LP relaxation
        min sum_{a!=b} p_ab f_ab
        s.t. f_ab + f_ba = 1               (each pair: one direction backward)
             f_ab + f_bc + f_ca >= 1       (no directed 3-cycle all-forward)
             0 <= f <= 1
    is a valid lower bound on returns, tighter than the 2-cycle bound. NP-hard MFAS
    itself is not needed; the LP suffices and is polynomial.
    """
    import itertools
    import numpy as np
    from scipy.optimize import linprog

    K = list(range(1, inst["n"] + 1))
    pairs = [(a, b) for a in K for b in K if a != b]
    idx = {pr: i for i, pr in enumerate(pairs)}
    nv = len(pairs)
    out = {}
    for t in inst["T"]:
        p = inst["p"][t]
        c = np.array([p[a][b] for (a, b) in pairs], dtype=float)
        A_eq, b_eq = [], []
        for a in K:
            for b in K:
                if a < b:
                    row = np.zeros(nv)
                    row[idx[(a, b)]] = 1.0
                    row[idx[(b, a)]] = 1.0
                    A_eq.append(row); b_eq.append(1.0)
        A_ub, b_ub = [], []
        for a, b, cc in itertools.combinations(K, 3):
            for x, y, zz in ((a, b, cc), (a, cc, b)):   # both directed 3-cycles
                row = np.zeros(nv)
                row[idx[(x, y)]] = -1.0
                row[idx[(y, zz)]] = -1.0
                row[idx[(zz, x)]] = -1.0
                A_ub.append(row); b_ub.append(-1.0)
        res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                      A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                      bounds=(0, 1), method="highs")
        if res.success:
            out[t] = round(res.fun, 6)
        else:  # fall back to the 2-cycle bound
            out[t] = sum(min(p[a][b], p[b][a]) for a in K for b in K if a < b)
    return out
