"""
Fast in-process version of flow_eval.eval_flow using scipy.linprog (HiGHS).

Same model (service-aware flow subproblem for fixed routing) as flow_eval.py, but
built as sparse LP matrices and solved in-process -- no CBC subprocess per call.
This is what makes the ALNS matheuristic and the scaled studies tractable
(~1-5 ms/eval vs ~70 ms for the PuLP+CBC path). Validated to match the PuLP
evaluator exactly (see tests in the scaling/validation scripts).
"""

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from flow_eval import route_arcs_by_period


def eval_flow_fast(inst, routes_by_period, mu, return_detail=False):
    n, T = inst["n"], inst["T"]
    K = list(range(1, n + 1))
    V = list(range(n + 1))
    d = inst["d"]
    Q, H = inst["Q"], inst["H"]
    arcs = route_arcs_by_period(routes_by_period)
    distance = sum(d[(i, j)] for t in T for (i, j) in arcs[t])

    # ---- variable indexing ----
    zidx = {}
    for t in T:
        for (i, j) in arcs[t]:
            for k in K:
                zidx[(t, i, j, k)] = len(zidx)
    nz = len(zidx)
    yidx = {}
    for t in T:
        for k in K:
            yidx[(t, k)] = nz + len(yidx)
    nvars = nz + len(yidx)

    dout = {t: {i: [j for (ii, j) in arcs[t] if ii == i] for i in V} for t in T}
    din = {t: {i: [j for (j, ii) in arcs[t] if ii == i] for i in V} for t in T}

    # objective: minimize mu * total returns (arcs ending at depot)
    c = np.zeros(nvars)
    for (t, i, j, k), col in zidx.items():
        if j == 0:
            c[col] = mu

    er, ec, ev, beq = [], [], [], []   # equality rows
    ur, uc, uv, bub = [], [], [], []   # inequality rows (<=)

    def eq_row():
        return len(beq)

    def ub_row():
        return len(bub)

    # (5) inflow_k - outflow_k - y_k = 0
    for t in T:
        for k in K:
            r = eq_row(); beq.append(0.0)
            for i in din[t][k]:
                er.append(r); ec.append(zidx[(t, i, k, k)]); ev.append(1.0)
            for j in dout[t][k]:
                er.append(r); ec.append(zidx[(t, k, j, k)]); ev.append(-1.0)
            er.append(r); ec.append(yidx[(t, k)]); ev.append(-1.0)
    # (6) generation of commodity k at origin j (j != k)
    for t in T:
        for j in K:
            for k in K:
                if j == k:
                    continue
                r = eq_row(); beq.append(float(inst["p"][t][j][k]))
                for i in dout[t][j]:
                    er.append(r); ec.append(zidx[(t, j, i, k)]); ev.append(1.0)
                for i in din[t][j]:
                    er.append(r); ec.append(zidx[(t, i, j, k)]); ev.append(-1.0)
    # (9)/(10) depot dispatch definition
    for idx, t in enumerate(T):
        for k in K:
            r = eq_row(); beq.append(float(inst["g"][t][k]))
            for j in dout[t][0]:
                er.append(r); ec.append(zidx[(t, 0, j, k)]); ev.append(1.0)
            if idx > 0:                       # subtract previous returns (10)
                tp = T[idx - 1]
                for i in din[tp][0]:
                    er.append(r); ec.append(zidx[(tp, i, 0, k)]); ev.append(-1.0)

    # (7) arc capacity: sum_k z_{t,i,j,k} <= Q
    for t in T:
        for (i, j) in arcs[t]:
            r = ub_row(); bub.append(float(Q))
            for k in K:
                ur.append(r); uc.append(zidx[(t, i, j, k)]); uv.append(1.0)
    # (8) y_k <= H
    for t in T:
        for k in K:
            r = ub_row(); bub.append(float(H))
            ur.append(r); uc.append(yidx[(t, k)]); uv.append(1.0)
    # (11) disp_k - y_k <= 0
    for t in T:
        for k in K:
            r = ub_row(); bub.append(0.0)
            for j in dout[t][0]:
                ur.append(r); uc.append(zidx[(t, 0, j, k)]); uv.append(1.0)
            ur.append(r); uc.append(yidx[(t, k)]); uv.append(-1.0)

    A_eq = coo_matrix((ev, (er, ec)), shape=(len(beq), nvars))
    A_ub = coo_matrix((uv, (ur, uc)), shape=(len(bub), nvars))
    res = linprog(c, A_ub=A_ub, b_ub=np.array(bub), A_eq=A_eq, b_eq=np.array(beq),
                  bounds=(0, None), method="highs")
    if not res.success:
        return {"feasible": False, "R": None, "distance": distance,
                "objective": float("inf")}
    xopt = res.x
    R = sum(xopt[zidx[(t, i, 0, k)]] for t in T for i in din[t][0] for k in K)
    R = round(R)
    out = {"feasible": True, "R": R, "distance": distance,
           "objective": round(distance + mu * R, 4)}
    if return_detail:
        out["returns_by_period"] = {
            t: {k: round(sum(xopt[zidx[(t, i, 0, k)]] for i in din[t][0])) for k in K}
            for t in T}
        out["y_by_period"] = {t: {k: round(xopt[yidx[(t, k)]]) for k in K} for t in T}
    return out
