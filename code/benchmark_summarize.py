"""
Turn the raw 48-instance benchmark run into an honest, publication-ready summary,
separating the reliable regime (n<=15, open-source solvable) from the tractability
frontier (n>=20, scenario B), where neither the exact open-source MILP nor the
matheuristic's construction yields a certified/feasible routing at the stated
capacities. Writes results/benchmark_summary.json and a LaTeX table.
"""
import json
import os
import statistics as st

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))

with open(os.path.join(RES, "benchmark_david.json")) as f:
    data = json.load(f)
rows = data["rows"]

# A row's exact result is TRUSTWORTHY only if proven optimal, or a valid integer
# incumbent (integer objective AND not flagged above the heuristic UB).
def exact_trustworthy(r):
    if r["cbc_flagged"]:
        return False
    o = r["exact_obj"]
    return abs(o - round(o)) < 1e-6      # integer objective => a real routing


reliable, frontier = [], []
for r in rows:
    r = dict(r)
    r["exact_ok"] = exact_trustworthy(r)
    r["heur_ok"] = r["heur_obj"] is not None
    (reliable if r["n"] <= 15 else frontier).append(r)

# per-size table over the reliable regime
by_size = {}
for r in reliable:
    by_size.setdefault(r["n"], []).append(r)

table = []
for n, sub in sorted(by_size.items()):
    opt = [x for x in sub if x["proven_optimal"]]
    et = [x["exact_time_s"] for x in sub if x["exact_ok"]]
    ht = [x["heur_time_s"] for x in sub if x["heur_ok"]]
    hv = [x["heur_vs_exact_pct"] for x in sub
          if x["heur_vs_exact_pct"] is not None]
    table.append({
        "n": n, "n_instances": len(sub),
        "n_proven_optimal": len(opt),
        "mean_exact_time_s": round(st.mean(et), 1) if et else None,
        "max_exact_time_s": round(max(et), 1) if et else None,
        "mean_heur_time_s": round(st.mean(ht), 1) if ht else None,
        "mean_heur_gap_pct": round(st.mean(hv), 2) if hv else 0.0,
        "max_heur_gap_pct": round(max(hv), 2) if hv else 0.0,
    })

# frontier characterization
frontier_summary = {
    "n_instances": len(frontier),
    "sizes": sorted(set(r["n"] for r in frontier)),
    "n_exact_certified": sum(r["exact_ok"] for r in frontier),
    "n_heuristic_feasible": sum(r["heur_ok"] for r in frontier),
    "note": ("scenario-B instances (n>=20): neither the open-source exact MILP "
             "(CBC/HiGHS return no certified integer solution within the limit) "
             "nor the matheuristic construction yields a flow-feasible routing at "
             "the stated Q=125, H=30. The complete-graph flow relaxation is "
             "feasible, so infeasibility is not established; demand-ordered "
             "construction removes the H-coupling barrier but a single ordered "
             "tour needs Q~276 (>2x the available 125), and splitting to respect "
             "Q reintroduces cross-tour deferral. Certifying these sizes needs a "
             "commercial solver (Gurobi, as used for the case study)."),
}

out = {
    "reliable_regime": {"sizes": "5-15", "per_size_table": table},
    "frontier": frontier_summary,
    "headline": {
        "exact_optimal_up_to_n": max((t["n"] for t in table
                                      if t["n_proven_optimal"] > 0), default=None),
        "all_optimal_up_to_n": max((t["n"] for t in table
                                    if t["n_proven_optimal"] == t["n_instances"]),
                                   default=None),
        "max_heur_gap_pct_reliable": max((t["max_heur_gap_pct"] for t in table),
                                         default=None),
        "mean_heur_time_s_at_15": next((t["mean_heur_time_s"] for t in table
                                        if t["n"] == 15), None),
        "mean_exact_time_s_at_15": next((t["mean_exact_time_s"] for t in table
                                         if t["n"] == 15), None),
    },
}
with open(os.path.join(RES, "benchmark_summary.json"), "w") as f:
    json.dump(out, f, indent=2)

# LaTeX table (reliable regime)
lines = [
    r"\begin{tabular}{r r r r r r r}",
    r"\toprule",
    r"$n$ & \#inst & proven opt. & mean exact (s) & max exact (s) & "
    r"mean heur.\ gap (\%) & max heur.\ gap (\%) \\",
    r"\midrule",
]
for t in table:
    lines.append(
        f"{t['n']} & {t['n_instances']} & {t['n_proven_optimal']} & "
        f"{t['mean_exact_time_s']} & {t['max_exact_time_s']} & "
        f"{t['mean_heur_gap_pct']} & {t['max_heur_gap_pct']} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
with open(os.path.join(RES, "benchmark_table.tex"), "w") as f:
    f.write("\n".join(lines))

print(json.dumps(out, indent=2))
print("\n--- LaTeX table ---\n" + "\n".join(lines))
