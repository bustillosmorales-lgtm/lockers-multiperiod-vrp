"""
Generality of the identifiability framework, on a SECOND, classic model unrelated to
the locker VRP: the capacitated facility location problem (CFLP). This shows the
identifiability of reported operational indicators is not a locker-routing curiosity
but a general property of design-vs-recourse optimization.

Model: candidate facilities j with opening cost o_j and capacity cap_j; customers i
with unit demand; assignment cost c_ij. Decisions: open_j in {0,1} (design, priced),
assignment y_ij >= 0 (recourse). Objective:  min sum_j o_j open_j + sum_ij c_ij y_ij.

Reported KPI (the kind managers publish): the load of a designated facility,
g = sum_i y_{i,1}  ("how many customers does facility 1 serve"). It is linear in the
recourse and NOT priced beyond total cost, so by the same argument as the locker rho
it can be unidentified.

We compute, exactly as for rho:
  z*        = optimal total cost (MILP);
  inner     = min/max g over the flow polytope at ONE optimal opening, holding total
              cost = z* (two LPs -- the assignment polytope is integral);
  optface   = min/max g over ALL cost-optimal (open, y) (a MILP -- the mixed-integer
              cost-optimal set of Proposition prop:optface).
inner subset optface always; strictly wider when alternative optimal openings differ.

Output: results/identifiability_general.json
"""
import json
import os

import pulp


def make(m, n, seed):
    """Deterministic pseudo-random CFLP on a grid (no RNG: reproducible closed form)."""
    # facility and customer coordinates on a small integer grid from the seed
    def coord(k, salt):
        v = (k * 2654435761 + salt * 40503 + seed * 97) % 1000
        return (v % 25, (v // 25) % 25)
    fac = [coord(j, 11) for j in range(m)]
    cus = [coord(i, 7) for i in range(n)]
    o = [12 for j in range(m)]                                  # equal opening costs -> tie-prone openings
    cap = [(n // m) + 1 + (j + seed) % 3 for j in range(m)]     # tight-ish capacities
    c = [[abs(fac[j][0] - cus[i][0]) + abs(fac[j][1] - cus[i][1])  # Manhattan (integer -> ties)
          for j in range(m)] for i in range(n)]
    return {"m": m, "n": n, "o": o, "cap": cap, "c": c}


def _base(prob, inst, relax_open, relax_y=True):
    m, n, c, cap = inst["m"], inst["n"], inst["c"], inst["cap"]
    ocat = "Continuous" if relax_open else "Binary"
    open_ = {j: pulp.LpVariable(f"open_{j}", lowBound=0, upBound=1, cat=ocat) for j in range(m)}
    y = {(i, j): pulp.LpVariable(f"y_{i}_{j}", lowBound=0,
                                 cat="Continuous" if relax_y else "Binary")
         for i in range(n) for j in range(m)}
    for i in range(n):
        prob += pulp.lpSum(y[(i, j)] for j in range(m)) == 1            # each customer served
    for j in range(m):
        prob += pulp.lpSum(y[(i, j)] for i in range(n)) <= cap[j] * open_[j]  # capacity + linking
    cost = (pulp.lpSum(inst["o"][j] * open_[j] for j in range(m))
            + pulp.lpSum(c[i][j] * y[(i, j)] for i in range(n) for j in range(m)))
    load1 = pulp.lpSum(y[(i, 0)] for i in range(n))   # reported KPI g = load of facility "1" (index 0)
    return open_, y, cost, load1


def optimal_cost(inst):
    prob = pulp.LpProblem("cflp", pulp.LpMinimize)
    open_, y, cost, _ = _base(prob, inst, relax_open=False, relax_y=False)
    prob += cost
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        return None, None
    opens = {j: int(round(open_[j].value())) for j in range(inst["m"])}
    return round(pulp.value(cost)), opens


def kpi_extreme(inst, zstar, maximize, fix_open=None):
    """min/max g s.t. total cost = z*. fix_open pins the opening (inner bound, LP);
    fix_open=None ranges over all openings (optimal face, MILP)."""
    sense = pulp.LpMaximize if maximize else pulp.LpMinimize
    prob = pulp.LpProblem("kpi", sense)
    relax_open = fix_open is not None
    open_, y, cost, load1 = _base(prob, inst, relax_open=relax_open, relax_y=True)
    prob += load1
    prob += cost == zstar                                              # stay cost-optimal
    if fix_open is not None:
        for j in range(inst["m"]):
            prob += open_[j] == fix_open[j]
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        return None
    return round(pulp.value(load1), 4)


def run_instance(m, n, seed):
    inst = make(m, n, seed)
    zstar, opens = optimal_cost(inst)
    if zstar is None:
        return None
    lo_in = kpi_extreme(inst, zstar, False, fix_open=opens)
    hi_in = kpi_extreme(inst, zstar, True, fix_open=opens)
    lo_full = kpi_extreme(inst, zstar, False, fix_open=None)
    hi_full = kpi_extreme(inst, zstar, True, fix_open=None)
    if None in (lo_in, hi_in, lo_full, hi_full):
        return None
    return {"m": m, "n": n, "seed": seed, "z_star": zstar,
            "inner_[lo,hi]": [lo_in, hi_in], "inner_width": round(hi_in - lo_in, 4),
            "optface_[lo,hi]": [lo_full, hi_full], "optface_width": round(hi_full - lo_full, 4),
            "kpi_unidentified": hi_in - lo_in > 1e-6,
            "optface_wider": (lo_full < lo_in - 1e-6) or (hi_full > hi_in + 1e-6)}


def main():
    rows = []
    for seed in range(24):
        r = run_instance(5, 14, seed)
        if r:
            rows.append(r)
            print(f"seed {seed}: z*={r['z_star']}  inner {r['inner_[lo,hi]']} (w{r['inner_width']})  "
                  f"optface {r['optface_[lo,hi]']} (w{r['optface_width']})  "
                  f"unident={r['kpi_unidentified']} wider={r['optface_wider']}", flush=True)
    unid = [r for r in rows if r["kpi_unidentified"]]
    wider = [r for r in rows if r["optface_wider"]]
    summary = {
        "model": "capacitated facility location (CFLP)",
        "kpi": "load of a designated facility (sum_i y_{i,1})",
        "n_instances": len(rows),
        "n_kpi_unidentified": len(unid),
        "n_optface_wider_than_inner": len(wider),
        "max_inner_width": round(max((r["inner_width"] for r in rows), default=0), 4),
        "max_optface_width": round(max((r["optface_width"] for r in rows), default=0), 4),
        "example": max(rows, key=lambda r: r["optface_width"], default=None),
    }
    outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    with open(os.path.join(outdir, "identifiability_general.json"), "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
