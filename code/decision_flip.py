"""
Consequence of non-identification: a decision taken from a reported indicator can be
solver-determined rather than model-determined. We make this concrete on the CFLP of
identifiability_general.py.

Decision: "which facility is the busiest" (e.g. the one to expand). For each facility
j we compute the range of its load over the ENTIRE cost-optimal set (the mixed-integer
optimal face of Proposition prop:optface). If two facilities' load ranges overlap so
that each can be the maximum at some cost-optimal solution, then the "busiest facility"
-- and hence the expansion decision -- is fixed by the solver's tie-break, not by the
model. We exhibit an instance where the argmax-load facility differs between two
cost-optimal solutions: a genuine decision flip at zero cost difference.

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
    return {"m": m, "n": n, "seed": seed, "z_star": zstar,
            "load_ranges": ranges,
            "strict_ranking_reversal": reversal,
            "decision_flips": reversal is not None}


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
    summary = {
        "model": "CFLP; decision = which facility is busiest (e.g. expand)",
        "n_instances": len(rows),
        "n_with_decision_flip": len(flips),
        "example": flips[0] if flips else None,
    }
    outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    with open(os.path.join(outdir, "decision_flip.json"), "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
