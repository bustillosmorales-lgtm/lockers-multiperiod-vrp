"""
ALNS matheuristic for the multi-period locker VRP with recirculation.

Design (the algorithmic contribution): the outer metaheuristic searches over
ROUTING only; every candidate routing is scored EXACTLY by the flow LP
(flow_eval.eval_flow), which returns the optimal service-aware flow dist + mu*R
for that routing. Thus the routing is heuristic but the flow/delivery/return
decision -- the part that Phase A showed is where the model's degeneracy and the
service trade-off live -- is solved to optimality at every step.

Routing search:
  - construction: nearest-neighbor giant tour per period (add vehicles until the
    flow LP is feasible);
  - local search: 2-opt and or-opt (distance-guided, cheap);
  - ALNS: random-removal destroy + greedy (distance) reinsertion for diversification;
  - selection: candidates scored by the EXACT objective via eval_flow; record-to-
    record acceptance; best-so-far retained.

Deterministic given (instance, seed): uses a seeded RNG; no wall-clock in logic.
"""

import random

from flow_eval_fast import eval_flow_fast as eval_flow


# ---------- routing helpers (single or multiple giant tours per period) --------
def _tour_distance(inst, tour):
    d = inst["d"]
    return sum(d[(tour[i], tour[i + 1])] for i in range(len(tour) - 1))


def _nn_tour(inst, start_pool, rng):
    """Nearest-neighbor tour [0, ...pool..., 0] over the given locker pool."""
    d = inst["d"]
    pool = list(start_pool)
    tour = [0]
    cur = 0
    while pool:
        nxt = min(pool, key=lambda k: d[(cur, k)])
        tour.append(nxt)
        pool.remove(nxt)
        cur = nxt
    tour.append(0)
    return tour


def _two_opt(tour, rng):
    """One random 2-opt segment reversal (returns a new tour)."""
    if len(tour) <= 4:
        return tour[:]
    i = rng.randint(1, len(tour) - 3)
    j = rng.randint(i + 1, len(tour) - 2)
    return tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]


def _or_opt(tour, rng):
    """Move a short segment (len 1-2) to another position."""
    inner = tour[1:-1]
    if len(inner) < 3:
        return tour[:]
    seglen = rng.choice([1, 2])
    i = rng.randint(0, len(inner) - seglen)
    seg = inner[i:i + seglen]
    rest = inner[:i] + inner[i + seglen:]
    j = rng.randint(0, len(rest))
    new_inner = rest[:j] + seg + rest[j:]
    return [0] + new_inner + [0]


def _destroy_repair(routes, inst, rng, q):
    """Remove q lockers at random from the period's routes and greedily
    reinsert them at the cheapest (distance) feasible position."""
    d = inst["d"]
    pool = [k for r in routes for k in r[1:-1]]
    if q >= len(pool):
        q = max(1, len(pool) // 3)
    removed = rng.sample(pool, q)
    new_routes = [[x for x in r if x not in removed] for r in routes]
    new_routes = [r for r in new_routes if len(r) > 2] or [[0, 0]]
    for k in removed:
        best = None
        for ri, r in enumerate(new_routes):
            for pos in range(1, len(r)):
                delta = (d[(r[pos - 1], k)] + d[(k, r[pos])]
                         - d[(r[pos - 1], r[pos])])
                if best is None or delta < best[0]:
                    best = (delta, ri, pos)
        _, ri, pos = best
        new_routes[ri] = new_routes[ri][:pos] + [k] + new_routes[ri][pos:]
    return new_routes


def _perturb(sol, inst, rng, max_vehicles):
    """Produce a neighbor routing by applying an operator to one random period."""
    t = rng.choice(list(sol.keys()))
    routes = [r[:] for r in sol[t]]
    op = rng.random()
    if op < 0.4 and routes:
        ri = rng.randrange(len(routes))
        routes[ri] = _two_opt(routes[ri], rng)
    elif op < 0.75 and routes:
        ri = rng.randrange(len(routes))
        routes[ri] = _or_opt(routes[ri], rng)
    else:
        q = rng.randint(1, max(1, inst["n"] // 3))
        routes = _destroy_repair(routes, inst, rng, q)
    new = {tt: [r[:] for r in sol[tt]] for tt in sol}
    new[t] = routes
    return new


def matheuristic(inst, mu, iters=600, seed=0, verbose=False, init_routes=None):
    """ALNS matheuristic. Returns best objective, routing, distance, R, and the
    number of exact flow evaluations used.

    init_routes: optional {t: [routes]} to warm-start from (e.g. the myopic
    solution). If given and feasible, the search starts there instead of the
    nearest-neighbour construction, guaranteeing the returned objective is no
    worse than the warm start."""
    rng = random.Random(seed)
    max_veh = inst.get("C", 2)

    # --- initial feasible solution
    best_sol = None
    if init_routes is not None:
        ev = eval_flow(inst, init_routes, mu)
        if ev["feasible"]:
            best_sol = {t: [r[:] for r in init_routes[t]] for t in inst["T"]}
            best = ev
    if best_sol is None:                 # nearest-neighbour construction fallback
        for veh in range(1, max_veh + 1):
            sol = {}
            for t in inst["T"]:
                groups = [[] for _ in range(veh)]
                order = sorted(range(1, inst["n"] + 1), key=lambda k: inst["d"][(0, k)])
                for idx, k in enumerate(order):
                    groups[idx % veh].append(k)
                sol[t] = [_nn_tour(inst, g, rng) for g in groups if g]
            ev = eval_flow(inst, sol, mu)
            if ev["feasible"]:
                best_sol = sol
                best = ev
                break
    if best_sol is None:
        return {"feasible": False}

    evals = 1
    cur_sol, cur = best_sol, best
    no_improve = 0
    for it in range(iters):
        cand = _perturb(cur_sol, inst, rng, max_veh)
        ev = eval_flow(inst, cand, mu)
        evals += 1
        if not ev["feasible"]:
            no_improve += 1
            continue
        # record-to-record acceptance around the best
        if ev["objective"] < cur["objective"] or \
           ev["objective"] <= best["objective"] * 1.02:
            cur_sol, cur = cand, ev
        if ev["objective"] < best["objective"]:
            best_sol, best = cand, ev
            no_improve = 0
            if verbose:
                print(f"  it={it} obj={best['objective']} dist={best['distance']} R={best['R']}")
        else:
            no_improve += 1
        if no_improve > iters // 3:          # soft restart from best
            cur_sol, cur = best_sol, best
            no_improve = 0

    return {"feasible": True, "objective": best["objective"],
            "distance": best["distance"], "R": best["R"],
            "routes": best_sol, "evals": evals}
