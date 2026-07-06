"""
Scaling study: the matheuristic vs the exact tractability wall.

For each network size:
  - matheuristic objective and runtime (fast, in-process flow LPs);
  - exact CBC objective within a time budget, flagged PROVEN (finished before the
    limit) or INCUMBENT (hit the limit -> not a proven optimum);
  - comparison: at sizes CBC proves optimality, the matheuristic gap to the
    proven optimum; beyond that, the matheuristic's improvement over CBC's
    time-limited incumbent.

Result: the matheuristic matches the proven optimum where CBC can prove it, and
increasingly dominates CBC's time-limited incumbent as the network grows past the
~12-15 locker wall, in seconds. (An LP-relaxation lower bound is also recorded,
but it is loose -- typical for VRP formulations -- so a tight bound via column
generation is the acknowledged next step. Production runs would use Gurobi.)

Run:  python scaling.py
"""

import csv
import json
import os
import time

import voi as V
import model_full as M
import matheuristic as H

MU = 5
TAU = 2
EXACT_TL = 60
EXACT_UP_TO = 15          # do not spend time on exact beyond this size
SIZES = [5, 8, 10, 12, 15, 20, 25, 30]


def make(n, seed=0):
    ref = V.total_flow(V.make_instance(n, TAU, seed, Q=999))
    Q = max(40, int(1.3 * ref))   # loose enough to stay feasible at every size
    inst = V.make_instance(n, TAU, seed, Q=Q)
    inst["C"] = 4
    return inst


def main():
    rows = []
    for n in SIZES:
        inst = make(n)
        t0 = time.time(); he = H.matheuristic(inst, MU, iters=600, seed=7); t_he = time.time() - t0
        if not he.get("feasible"):
            print(f"n={n:>2} matheuristic INFEASIBLE"); continue
        lb = M.lower_bound(inst, MU, time_limit=45)

        ex_obj = ex_time = None; ex_proven = None
        if n <= EXACT_UP_TO:
            t0 = time.time(); ex = M.solve_joint(inst, MU, time_limit=EXACT_TL); ex_time = time.time() - t0
            ex_obj = ex.get("objective")
            ex_proven = ex_time < EXACT_TL - 3        # finished before the limit

        gap_proven = (round(100 * (he["objective"] - ex_obj) / ex_obj, 2)
                      if ex_proven and ex_obj else None)
        heur_vs_incumbent = (round(100 * (he["objective"] - ex_obj) / ex_obj, 2)
                             if (ex_obj is not None and not ex_proven) else None)
        row = {"n": n, "Q": inst["Q"], "heur_obj": he["objective"],
               "heur_time_s": round(t_he, 1), "evals": he.get("evals"),
               "LB_lp": lb, "exact_obj": ex_obj,
               "exact_time_s": round(ex_time, 1) if ex_time else None,
               "exact_proven": ex_proven,
               "gap_to_proven_%": gap_proven,
               "heur_vs_incumbent_%": heur_vs_incumbent}
        rows.append(row)
        tag = (f"proven opt, gap={gap_proven}%" if ex_proven
               else f"incumbent, heur {heur_vs_incumbent}% vs it" if ex_obj is not None
               else "exact skipped (intractable)")
        print(f"n={n:>2} heur={he['objective']:>7} ({t_he:>4.1f}s)  exact={ex_obj} "
              f"[{'PROVEN' if ex_proven else 'INCUMBENT' if ex_obj else '-'} "
              f"{ex_time:.0f}s]  {tag}" if ex_time else
              f"n={n:>2} heur={he['objective']:>7} ({t_he:>4.1f}s)  {tag}")

    outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    with open(os.path.join(outdir, "scaling.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    with open(os.path.join(outdir, "scaling.json"), "w") as f:
        json.dump({"mu": MU, "tau": TAU, "exact_time_limit_s": EXACT_TL, "rows": rows}, f, indent=2)
    print(f"\nWritten scaling.csv / scaling.json in {outdir}")


if __name__ == "__main__":
    main()
