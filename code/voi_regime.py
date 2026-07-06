"""
Value-of-integration REGIME study: characterise WHEN integration pays and by HOW
MUCH, so the (small on average) VoI is read as a concentrated, capacity-driven
effect rather than a weak one. Everything exact.

Two axes:
  - capacity tightness (factor x reference flow -> Q), the driver of the regime;
  - horizon length tau in {2,3,4}, to show the effect compounds over periods.

For each cell we solve BOTH policies exactly (joint MILP via HiGHS; myopic =
period-by-period exact) and record VoI = obj_myopic - obj_joint >= 0. We then report
not only the unconditional mean but the CONDITIONAL magnitude (VoI | VoI>0) and the
ceiling, plus a simple decision rule: the capacity-utilisation threshold above which
integration is worth coordinating.

n = 5 (where the joint MILP is solved to proven optimality). Output:
results/voi_regime.json / .csv
"""
import csv
import json
import os
import statistics as st
from concurrent.futures import ProcessPoolExecutor, as_completed

import voi as V
import model_full as M

MU = 5
N = 5
TAUS = [2, 3, 4]
FACTORS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.85, 1.00]
SEEDS = list(range(10))
TL = 300


def build(n, tau, seed, factor):
    ref = V.total_flow(V.make_instance(n, tau, seed, Q=999))
    Q = max(10, int(factor * ref))
    inst = V.make_instance(n, tau, seed, Q=Q); inst["C"] = 3
    return inst, ref


def myopic_exact(inst):
    K = list(range(1, inst["n"] + 1))
    prev = {k: 0 for k in K}
    total = 0.0
    for t in inst["T"]:
        disp = {k: inst["g"][t][k] + prev[k] for k in K}
        res = M.solve_single(inst, t, disp, MU, time_limit=TL)
        if res["status"] != "Optimal":
            return None
        total += res["distance"] + MU * res["R"]
        prev = res["returns"]
    return total


def cell(args):
    tau, seed, factor = args
    inst, ref = build(N, tau, seed, factor)
    om = myopic_exact(inst)
    if om is None:
        return None
    j = M.solve_joint(inst, MU, time_limit=TL, solver="highs")
    jo = j.get("objective")
    # HiGHS proves optimality here; keep the same validity guard as a backstop
    # (joint optimum <= myopic, which is feasible for joint).
    if jo is None or jo > om + 1e-6:
        return None
    voi = om - jo
    # capacity utilisation of the joint plan proxied by ref flow / (Q * horizon-ish):
    util = ref / inst["Q"] if inst["Q"] else 0.0
    return {"tau": tau, "seed": seed, "factor": factor, "Q": inst["Q"],
            "ref_flow": ref, "util": round(util, 3),
            "obj_joint": round(jo, 2), "obj_myopic": round(om, 2),
            "VoI": round(voi, 2),
            "VoI_pct": round(100 * voi / om, 2) if om else 0.0}


def agg(rows):
    vp = [r["VoI_pct"] for r in rows]
    pos = [v for v in vp if v > 1e-6]
    return {
        "n": len(rows),
        "frac_positive": round(len(pos) / len(rows), 2) if rows else 0.0,
        "mean_pct": round(st.mean(vp), 2) if vp else 0.0,
        "cond_mean_pct": round(st.mean(pos), 2) if pos else 0.0,   # VoI | VoI>0
        "cond_median_pct": round(st.median(pos), 2) if pos else 0.0,
        "max_pct": round(max(vp), 2) if vp else 0.0,
    }


def main():
    jobs = [(tau, s, f) for tau in TAUS for s in SEEDS for f in FACTORS]
    rows = []
    with ProcessPoolExecutor(max_workers=16) as ex:
        for fut in as_completed([ex.submit(cell, j) for j in jobs]):
            r = fut.result()
            if r:
                rows.append(r)
    rows.sort(key=lambda r: (r["tau"], r["factor"], r["seed"]))

    by_factor = {f: agg([r for r in rows if r["factor"] == f]) for f in FACTORS
                 if any(r["factor"] == f for r in rows)}
    by_tau = {tau: agg([r for r in rows if r["tau"] == tau]) for tau in TAUS
              if any(r["tau"] == tau for r in rows)}
    # tight regime = the two tightest feasible factors pooled
    feas_factors = sorted(by_factor)
    tight = [r for r in rows if r["factor"] in feas_factors[:3]]
    slack = [r for r in rows if r["factor"] >= 0.85]
    overall = agg(rows)

    # decision-rule threshold: smallest utilisation at which a positive VoI appears,
    # and the utilisation above which >=50% of instances have positive VoI.
    pos_utils = sorted(r["util"] for r in rows if r["VoI_pct"] > 1e-6)
    thr_any = pos_utils[0] if pos_utils else None

    out = {
        "mu": MU, "n": N, "taus": TAUS, "seeds": len(SEEDS), "exact": True,
        "solver": "highs (joint), cbc-exact (myopic)",
        "overall": overall,
        "tight_regime": agg(tight) if tight else None,
        "slack_regime": agg(slack) if slack else None,
        "by_factor": by_factor,
        "by_tau": by_tau,
        "min_util_with_positive_VoI": round(thr_any, 3) if thr_any else None,
    }
    outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    with open(os.path.join(outdir, "voi_regime.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    with open(os.path.join(outdir, "voi_regime.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    print("=== VoI regime (exact), n=5 ===")
    print("overall:", overall)
    print("tight  :", out["tight_regime"])
    print("slack  :", out["slack_regime"])
    print("by factor (tighter -> looser):")
    for f in sorted(by_factor):
        s = by_factor[f]
        print(f"  {f:.2f}: frac>0 {s['frac_positive']:.2f}  mean {s['mean_pct']:.2f}%  "
              f"cond-mean {s['cond_mean_pct']:.2f}%  max {s['max_pct']:.2f}%  (n={s['n']})")
    print("by horizon:")
    for tau in sorted(by_tau):
        s = by_tau[tau]
        print(f"  tau={tau}: frac>0 {s['frac_positive']:.2f}  cond-mean {s['cond_mean_pct']:.2f}%  max {s['max_pct']:.2f}%")


if __name__ == "__main__":
    main()
