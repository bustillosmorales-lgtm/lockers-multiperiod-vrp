"""
Within-instance rho comparison (fixes the confound in the rho-by-size figure:
the base dashed curve came from the original, unpublished benchmark, while the
identified curve is on regenerated instances -- two different instance sets).

Here, on the SAME regenerated instances, we compute BOTH:
  - the distance-only reading of rho: not a point but the interval [rho_min,
    rho_max] over the optimal flows of a min-distance routing (the non-identified
    set, exactly as in the certificate); and
  - the service-aware identified rho (mu=5).
So the flattening is measured within-instance and cannot be an artifact of
regeneration.

Output: results/within_instance_rho.json
Run:  python within_instance_rho.py
"""
import json
import os
import statistics as st

import voi as V
import matheuristic as H
from flow_eval_fast import eval_flow_fast

MU = 5
TAU = 2
SIZES = [5, 8, 10, 12, 15, 20]
SEEDS = [0, 1, 2]


def make(n, seed):
    ref = V.total_flow(V.make_instance(n, TAU, seed, Q=999))
    Q = max(20, int(0.6 * ref))          # tight-ish, where recirculation is active
    inst = V.make_instance(n, TAU, seed, Q=Q); inst["C"] = 3
    return inst


def rho_of(det):
    Y = sum(v for per in det["y_by_period"].values() for v in per.values())
    R = det["R"]
    return R / (Y + R) if (Y + R) > 0 else 0.0


def main():
    per_size = {n: {"lo": [], "hi": [], "ident": []} for n in SIZES}
    for n in SIZES:
        for seed in SEEDS:
            inst = make(n, seed)
            # min-distance routing (mu=0) -> non-identified interval on THIS instance
            base = H.matheuristic(inst, 0, iters=400, seed=seed)
            if not base["feasible"]:
                continue
            r = base["routes"]
            lo = rho_of(eval_flow_fast(inst, r, 1.0, return_detail=True))   # min returns
            hi = rho_of(eval_flow_fast(inst, r, -1.0, return_detail=True))  # max returns
            # service-aware identified rho on the SAME instance
            serv = H.matheuristic(inst, MU, iters=400, seed=seed)
            if not serv["feasible"]:
                continue
            ident = rho_of(eval_flow_fast(inst, serv["routes"], MU, return_detail=True))
            per_size[n]["lo"].append(lo); per_size[n]["hi"].append(hi)
            per_size[n]["ident"].append(ident)
            print(f"n={n:>2} seed={seed}: distance-only rho in [{lo:.3f},{hi:.3f}]  "
                  f"identified rho={ident:.3f}", flush=True)

    summary = {}
    for n in SIZES:
        d = per_size[n]
        if not d["ident"]:
            continue
        summary[n] = {"rho_min_mean": round(st.mean(d["lo"]), 4),
                      "rho_max_mean": round(st.mean(d["hi"]), 4),
                      "rho_identified_mean": round(st.mean(d["ident"]), 4),
                      "n_instances": len(d["ident"])}
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "results", "within_instance_rho.json"))
    with open(out, "w") as f:
        json.dump({"mu": MU, "tau": TAU, "by_size": summary}, f, indent=2)
    print("\n=== within-instance: distance-only interval vs identified rho ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
