"""
Benchmark the formulation on David's 48 real-geography scenario instances
(nueva info/instancias_txt/), derived from the Kandoo Guayaquil network.
For each instance, under the distance-only objective (mu=0):

  - EXACT: joint MILP via CBC with a time limit. Proven-optimal is detected by
    finish-before-limit (NOT the solver's status flag, which mislabels
    time-limited incumbents as "Optimal" -- the documented CBC gotcha). The
    matheuristic objective is used as a trusted upper bound: any exact incumbent
    above it is flagged.
  - HEURISTIC: ALNS matheuristic (routing search + exact flow LP scoring).
  - LP bound: LP relaxation of the joint MILP -> lower bound, for an optimality
    gap even when the MILP does not close.

Output: results/benchmark_david.{json,csv}
"""
import csv
import json
import os
import time

import david_data as D
import model_full as M
import matheuristic as H
from flow_eval_fast import eval_flow_fast as eval_flow

EXACT_TL = 90       # s, exact MILP time limit per instance
LP_TL = 30          # s, LP relaxation time limit
HEUR_ITERS = 500

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))

# scenario metadata (size, sweep dimension) from the filename / resumen csv
def _meta_from_name(name):
    # e.g. S15_A_SC026 -> size 15, scenario A, id SC026
    stem = name.replace(".txt", "")
    size = int(stem.split("_")[0][1:])
    scen = stem.split("_")[1]
    scid = stem.split("_")[2]
    return size, scen, scid


def run_instance(path):
    name = os.path.basename(path)
    size, scen, scid = _meta_from_name(name)
    inst = D.parse_instance_txt(path)

    # heuristic (trusted upper bound)
    t0 = time.time()
    h = H.matheuristic(inst, mu=0, iters=HEUR_ITERS, seed=0)
    t_h = time.time() - t0
    h_obj = h["objective"] if h.get("feasible") else None

    # LP relaxation lower bound
    try:
        lb = M.lower_bound(inst, mu=0, time_limit=LP_TL)
    except Exception:
        lb = None

    # exact MILP (time-based optimality detection)
    t0 = time.time()
    ex = M.solve_joint(inst, mu=0, time_limit=EXACT_TL, solver="cbc")
    t_ex = time.time() - t0
    ex_obj = ex["objective"]
    proven_opt = t_ex < EXACT_TL * 0.9        # finished before the limit
    # guard the CBC gotcha: never trust an exact incumbent above the heuristic UB
    flagged = h_obj is not None and ex_obj > h_obj + 1e-6

    best_ub = min([v for v in (ex_obj, h_obj) if v is not None])
    gap_lb = (round(100 * (best_ub - lb) / best_ub, 2)
              if lb and best_ub else None)
    heur_gap = (round(100 * (h_obj - ex_obj) / ex_obj, 2)
                if h_obj is not None and ex_obj and not flagged else None)

    row = {
        "instance": name, "n": size, "scenario": scen, "scid": scid,
        "C": inst["C"], "Q": inst["Q"], "H": inst["H"],
        "exact_obj": ex_obj, "exact_time_s": round(t_ex, 1),
        "proven_optimal": proven_opt, "cbc_flagged": flagged,
        "heur_obj": h_obj, "heur_time_s": round(t_h, 1), "heur_evals": h.get("evals"),
        "lp_lower_bound": lb, "gap_to_lb_pct": gap_lb,
        "heur_vs_exact_pct": heur_gap,
        "best_ub": best_ub,
    }
    return row


def main():
    paths = D.all_instance_paths()
    rows = []
    for i, path in enumerate(paths, 1):
        r = run_instance(path)
        rows.append(r)
        print(f"[{i:>2}/{len(paths)}] {r['instance']:<16} n={r['n']:>2} "
              f"exact={r['exact_obj']:>6} ({r['exact_time_s']:>5}s "
              f"{'OPT' if r['proven_optimal'] else 'inc'}"
              f"{' FLAG' if r['cbc_flagged'] else ''})  "
              f"heur={r['heur_obj']} ({r['heur_time_s']}s)  "
              f"gapLB={r['gap_to_lb_pct']}%  hVe={r['heur_vs_exact_pct']}%",
              flush=True)
        # incremental save
        with open(os.path.join(OUT, "benchmark_david.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader(); wr.writerows(rows)

    # summary by size
    by_size = {}
    for r in rows:
        by_size.setdefault(r["n"], []).append(r)
    size_summary = {}
    for n, sub in sorted(by_size.items()):
        opt = [x for x in sub if x["proven_optimal"]]
        hv = [x["heur_vs_exact_pct"] for x in sub if x["heur_vs_exact_pct"] is not None]
        size_summary[n] = {
            "n_instances": len(sub),
            "n_proven_optimal": len(opt),
            "mean_exact_time_s": round(sum(x["exact_time_s"] for x in sub) / len(sub), 1),
            "mean_heur_time_s": round(sum(x["heur_time_s"] for x in sub) / len(sub), 1),
            "mean_heur_vs_exact_pct": round(sum(hv) / len(hv), 2) if hv else None,
            "n_cbc_flagged": sum(x["cbc_flagged"] for x in sub),
        }

    out = {
        "config": {"exact_time_limit_s": EXACT_TL, "lp_time_limit_s": LP_TL,
                   "heur_iters": HEUR_ITERS, "objective": "distance-only (mu=0)",
                   "source": "David Kandoo-derived 48-instance benchmark"},
        "size_summary": size_summary, "rows": rows,
    }
    with open(os.path.join(OUT, "benchmark_david.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== size summary ===")
    print(json.dumps(size_summary, indent=2))
    print(f"\nWritten: {OUT}\\benchmark_david.json / .csv")


if __name__ == "__main__":
    main()
