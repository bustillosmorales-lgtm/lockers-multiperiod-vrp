"""
Consequence of non-identification: a decision taken from a reported indicator can be
solver-determined rather than model-determined. We make this concrete on the CFLP of
identifiability_general.py.

Decision: a pairwise ranking "is facility A busier than facility B" (which decides,
say, the expansion target between two candidates). For a pair (j,k) we compute the
range of load_j - load_k over the ENTIRE cost-optimal set (the mixed-integer optimal
face of Proposition prop:optface). If that range straddles zero -- load_j > load_k at
one cost-optimal solution and load_k > load_j at another -- the ranking, and the
decision keyed to it, is fixed by the solver's tie-break, not the model: a genuine
STRICT ranking reversal at zero cost difference. (Note: it is a pairwise ranking that
flips, not necessarily the global argmax/"busiest overall".)

Output: results/decision_flip.json
"""
import json
import os

import pulp

import identifiability_general as G


def load_extreme(inst, zstar, j, maximize):
    """min/max load of facility j over the full cost-optimal set (open binary)."""
    sense = pulp.LpMaximize if maximize else pulp.LpMinimize
    prob = pulp.LpProblem("loadj", sense)
    open_, y, cost, _ = G._base(prob, inst, relax_open=False, relax_y=True)
    loadj = pulp.lpSum(y[(i, j)] for i in range(inst["n"]))
    prob += loadj
    prob += cost == zstar
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        return None
    return round(pulp.value(loadj), 4)


def diff_config(inst, zstar, j, k, maximize):
    """A cost-optimal solution and the value of load_j - load_k on it, at the extreme."""
    sense = pulp.LpMaximize if maximize else pulp.LpMinimize
    prob = pulp.LpProblem("diff", sense)
    open_, y, cost, _ = G._base(prob, inst, relax_open=False, relax_y=True)
    diff = pulp.lpSum(y[(i, j)] - y[(i, k)] for i in range(inst["n"]))
    prob += diff
    prob += cost == zstar
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        return None
    loads = [round(sum(y[(i, jj)].value() for i in range(inst["n"])), 2) for jj in range(inst["m"])]
    return round(pulp.value(diff), 4), loads


def analyse(m, n, seed):
    inst = G.make(m, n, seed)
    zstar, _ = G.optimal_cost(inst)
    if zstar is None:
        return None
    ranges = {j: [load_extreme(inst, zstar, j, False), load_extreme(inst, zstar, j, True)]
              for j in range(m)}
    # STRICT ranking reversal: a pair (j,k) with load_j > load_k on one optimum and
    # load_k > load_j on another (the range of load_j - load_k over the optimal set
    # straddles 0). Then "is j busier than k" is not determined by the model.
    reversal = None
    for j in range(m):
        for k in range(j + 1, m):
            hi = diff_config(inst, zstar, j, k, True)
            lo = diff_config(inst, zstar, j, k, False)
            if hi is None or lo is None:
                continue
            if hi[0] > 1e-6 and lo[0] < -1e-6:      # j busier in one optimum, k in another
                reversal = {"pair": [j, k],
                            "loads_when_j_busier": hi[1], "loads_when_k_busier": lo[1]}
                break
        if reversal:
            break
    # service regret: a manager maximizing a facility's throughput (an unpriced
    # secondary payoff) can forgo up to (max load - min load) over the cost-optimal
    # set by trusting an arbitrary optimum. The largest such gap over facilities:
    widths = [ranges[j][1] - ranges[j][0] for j in range(m)
              if ranges[j][0] is not None and ranges[j][1] is not None]
    service_regret = round(max(widths), 4) if widths else 0.0
    return {"m": m, "n": n, "seed": seed, "z_star": zstar,
            "load_ranges": ranges,
            "strict_ranking_reversal": reversal,
            "decision_flips": reversal is not None,
            "service_regret_max_load_gap": service_regret}


def main():
    rows = []
    flips = []
    for seed in range(24):
        r = analyse(5, 14, seed)
        if r:
            rows.append(r)
            if r["decision_flips"]:
                rv = r["strict_ranking_reversal"]
                print(f"seed {seed}: STRICT REVERSAL pair {rv['pair']} -- "
                      f"j-busier {rv['loads_when_j_busier']} vs k-busier {rv['loads_when_k_busier']}", flush=True)
            else:
                print(f"seed {seed}: no strict reversal", flush=True)
            if r["decision_flips"]:
                flips.append(r)
    regrets = [r["service_regret_max_load_gap"] for r in rows]
    summary = {
        "model": "CFLP; decision = pairwise facility ranking (e.g. expansion target)",
        "n_instances": len(rows),
        "n_with_strict_ranking_reversal": len(flips),
        "max_service_regret_load_gap": round(max(regrets), 4) if regrets else 0.0,
        "mean_service_regret_load_gap": round(sum(regrets) / len(regrets), 4) if regrets else 0.0,
        "example": flips[0] if flips else None,
    }
    outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    with open(os.path.join(outdir, "decision_flip.json"), "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
