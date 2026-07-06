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


def argmax_config(inst, zstar, favour):
    """A cost-optimal solution that maximizes the load of facility `favour`; return the
    per-facility load vector and the argmax facility. Shows which facility a solver that
    happens to favour `favour` would report as busiest."""
    prob = pulp.LpProblem("cfg", pulp.LpMaximize)
    open_, y, cost, _ = G._base(prob, inst, relax_open=False, relax_y=True)
    prob += pulp.lpSum(y[(i, favour)] for i in range(inst["n"]))
    prob += cost == zstar
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        return None
    loads = [round(sum(y[(i, j)].value() for i in range(inst["n"])), 2) for j in range(inst["m"])]
    return loads, max(range(inst["m"]), key=lambda j: loads[j])


def analyse(m, n, seed):
    inst = G.make(m, n, seed)
    zstar, _ = G.optimal_cost(inst)
    if zstar is None:
        return None
    ranges = {j: [load_extreme(inst, zstar, j, False), load_extreme(inst, zstar, j, True)]
              for j in range(m)}
    # a "busiest facility" flip exists if some facility j can be strictly above every
    # other at one optimum yet another facility k can equal-or-exceed it at another.
    busiest = set()
    configs = {}
    for favour in range(m):
        res = argmax_config(inst, zstar, favour)
        if res:
            loads, am = res
            busiest.add(am)
            configs[favour] = {"loads": loads, "argmax_facility": am}
    return {"m": m, "n": n, "seed": seed, "z_star": zstar,
            "load_ranges": ranges,
            "distinct_busiest_facilities_across_optima": sorted(busiest),
            "decision_flips": len(busiest) > 1,
            "example_configs": configs}


def main():
    rows = []
    flips = []
    for seed in range(24):
        r = analyse(5, 14, seed)
        if r:
            rows.append(r)
            tag = "  <-- BUSIEST-FACILITY FLIP" if r["decision_flips"] else ""
            print(f"seed {seed}: busiest facilities over optima = "
                  f"{r['distinct_busiest_facilities_across_optima']}{tag}", flush=True)
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
