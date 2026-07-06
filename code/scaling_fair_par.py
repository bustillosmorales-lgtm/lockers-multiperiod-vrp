"""
Fair-budget scaling, PARALLEL. Each size's exact solve is independent, so we run
them concurrently across cores: wall-clock ~= the slowest single solve instead of
the sum. Same content as scaling_fair.py (CBC at a generous 600 s budget vs the
matheuristic), just parallelized.

Output: results/scaling_fair.json / .csv
Run:  python scaling_fair_par.py
"""
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import voi as V
import model_full as M
import matheuristic as H

MU = 5
TAU = 2
EXACT_TL = 600
SIZES = [5, 8, 10, 12, 15, 20]


def make(n, seed=7):
    ref = V.total_flow(V.make_instance(n, TAU, seed, Q=999))
    Q = max(40, int(1.3 * ref))
    inst = V.make_instance(n, TAU, seed, Q=Q); inst["C"] = 4
    return inst


def run_size(n):
    inst = make(n)
    t0 = time.time(); he = H.matheuristic(inst, MU, iters=600, seed=7); t_he = time.time() - t0
    t0 = time.time(); ex = M.solve_joint(inst, MU, time_limit=EXACT_TL); t_ex = time.time() - t0
    proven = ex["status"] == "Optimal" and t_ex < EXACT_TL - 5
    exo = ex.get("objective")
    vs = (round(100 * (he["objective"] - exo) / exo, 1) if exo else None)
    return {"n": n, "heur_obj": he["objective"], "heur_time_s": round(t_he, 1),
            "exact_obj": exo, "exact_time_s": round(t_ex, 1),
            "exact_proven": proven, "heur_vs_exact_%": vs}


def main():
    rows = []
    with ProcessPoolExecutor(max_workers=len(SIZES)) as ex:
        futs = {ex.submit(run_size, n): n for n in SIZES}
        for fut in as_completed(futs):
            r = fut.result()
            rows.append(r)
            print(f"n={r['n']:>2} heur={r['heur_obj']:>7} ({r['heur_time_s']:>4.1f}s)  "
                  f"exact={r['exact_obj']} [{'PROVEN' if r['exact_proven'] else 'incumbent'} "
                  f"{r['exact_time_s']:.0f}s]  heur vs exact = {r['heur_vs_exact_%']}%", flush=True)
    rows.sort(key=lambda r: r["n"])
    outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    with open(os.path.join(outdir, "scaling_fair.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    with open(os.path.join(outdir, "scaling_fair.json"), "w") as f:
        json.dump({"mu": MU, "tau": TAU, "exact_time_limit_s": EXACT_TL,
                   "solver": "CBC", "rows": rows}, f, indent=2)
    print("\nwritten scaling_fair.json")


if __name__ == "__main__":
    main()
