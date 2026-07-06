"""
Tighter lower bound via subtour-elimination cutting planes.

The plain LP relaxation of the base model is weak because subtours are ruled out
only by the (weak) MTZ-style load variables. We strengthen it by iteratively
adding directed subtour-elimination / connectivity cuts

    sum_{(i,j): i notin S, j in S} x_ij^t  >=  1     for all S subset of K,

separated exactly per period by a max-flow / min-cut between the depot and each
locker on the fractional support graph. Each added cut is a valid inequality, so
the resulting LP value is a valid lower bound -- much tighter than the plain
relaxation. (A full branch-and-price with route columns is the next step; this
cutting-plane bound is what we use to report optimality gaps at scale.)

Deps: pulp (CBC LP), networkx (min-cut).
"""

import networkx as nx
import pulp

from model_full import _add_routing_flow, _arcs


def lower_bound_cuts(inst, mu, max_rounds=20, time_limit=60, verbose=False):
    n, T = inst["n"], inst["T"]
    K = list(range(1, n + 1))
    d = inst["d"]
    A = _arcs(n)

    prob = pulp.LpProblem("lb_cuts", pulp.LpMinimize)
    blocks = {}
    prev = None
    dist_terms, ret_terms = [], []
    for idx, t in enumerate(T):
        disp = ({k: inst["g"][t][k] for k in K} if idx == 0
                else {k: inst["g"][t][k] + prev[k] for k in K})
        x, y, z, ret = _add_routing_flow(prob, inst, t, disp, mu, tag=f"lb{t}", relax=True)
        blocks[t] = (x, y, z, ret)
        prev = ret
        dist_terms.append(pulp.lpSum(d[(i, j)] * x[(i, j)] for (i, j) in A))
        ret_terms.append(pulp.lpSum(ret[k] for k in K))
    prob += pulp.lpSum(dist_terms) + mu * pulp.lpSum(ret_terms)

    value = None
    for rnd in range(max_rounds):
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))
        value = pulp.value(prob.objective)
        added = 0
        for t in T:
            x = blocks[t][0]
            G = nx.DiGraph()
            G.add_node(0)
            for (i, j) in A:
                val = x[(i, j)].value() or 0.0
                if val > 1e-7:
                    G.add_edge(i, j, capacity=float(val))
            seen = set()
            for k in K:
                if k not in G:
                    continue
                try:
                    cutval, (reach, nonreach) = nx.minimum_cut(G, 0, k)
                except Exception:
                    continue
                if cutval < 1.0 - 1e-4:
                    S = frozenset(set(nonreach) & set(K))
                    if not S or S in seen:
                        continue
                    seen.add(S)
                    prob += (pulp.lpSum(x[(i, j)] for (i, j) in A
                                        if i not in S and j in S) >= 1)
                    added += 1
        if verbose:
            print(f"  round {rnd}: LB={value:.1f}  cuts_added={added}")
        if added == 0:
            break
    return round(value, 4)
