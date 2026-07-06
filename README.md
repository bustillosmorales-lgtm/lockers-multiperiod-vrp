# Multi-period locker VRP — Q1 extension: reproducibility package

Extension of the multi-period one-to-one multi-commodity pickup-and-delivery
VRP with parcel lockers (De Santis, base manuscript `Versión0.tex`). Authors:
De Santis Bermeo (ESPOL) and F. Bustillos. This repository holds the theory
notes, code, and machine-checkable evidence behind the extended paper. The paper
links here for full reproducibility.

> Status: **Phase A** (ρ non-identifiability), **Phase B** (service-aware
> objective + value-of-integration), and **Phase C** (scalable matheuristic)
> complete. Next: tight lower bound (column generation) and full benchmark.

## Layout

```
q1-extension/
├── theory/
│   ├── identifiability_theory.md     General identifiability theorem (design vs recourse; rho is a corollary)
│   ├── rho_nonidentifiability.md     Phase A: result + proof (manuscript notation)
│   ├── voi_service_objective.md      Phase B: service objective + VoI regimes
│   └── matheuristic_scaling.md       Phase C: method + scaling study
├── code/
│   ├── certify_rho.py                Phase A certificate: both instances
│   ├── model_full.py                 Full multi-period MILP (2-14) + LP lower bound
│   ├── voi.py                        Phase B: joint vs myopic VoI sweep
│   ├── voi_tau3.py                   Horizon-length corroboration (tau=2 vs 3)
│   ├── flow_eval.py / flow_eval_fast.py   Exact flow evaluator (PuLP ref + scipy/HiGHS)
│   ├── matheuristic.py               Phase C: ALNS (routing search + exact flow, warm-start)
│   ├── bound_cuts.py                 Phase C: subtour-elimination cutting-plane lower bound
│   ├── branch_price.py               Phase C: column-generation (root) lower bound (MILP + bitmask DP pricer)
│   ├── bound_scale.py                Phase C: CG bound at n=10-15 via DP pricer -> bound_scale.json
│   ├── valid_ineq.py                 Phase C: return-flow valid inequalities (2- and 3-cycle)
│   ├── bound_ladder.py               Phase C: LP->cuts->CG->2cyc->3cyc gap ladder -> results/bound_ladder.json
│   ├── scaling.py                    Phase C: matheuristic vs exact wall (60 s baseline)
│   ├── scaling_fair_par.py           Phase C: fair-budget scaling (exact CBC 600 s, parallel)
│   ├── scaling_highs.py              Phase C: honest scaling vs HiGHS (strong open-source) -> scaling_highs.json
│   ├── scaling_large.py              Phase C: matheuristic at n=40-100 (HiGHS fails) -> scaling_large.json
│   ├── voi_scaled.py                 VoI at scale via matheuristic warm-started from myopic
│   ├── voi_controlled.py             VoI controlled: exact joint vs exact myopic, capacity sweep -> voi_controlled.json
│   ├── within_instance_rho.py        distance-only rho interval vs identified rho, same instances
│   ├── benchmark_service.py          Section 6 rho tables recomputed under the service objective (identified)
│   ├── real_benchmark.py             Real-geography instances: OSM parcel-locker coords + OSMnx road distances
│   ├── real_bench_run.py             rho non-identifiability + VoI on the real-geography benchmark -> results/real_benchmark.json
│   ├── voi_regime.py                 VoI magnitude regime (capacity x horizon, exact, conditional stats) -> results/voi_regime.json
│   ├── mu_sweep.py                   rho(mu) and distance(mu) trade-off sweep
│   ├── make_figure.py                renders figures/mu_tradeoff.pdf
│   ├── fig_mechanism.py              renders figures/mechanism.pdf (conceptual schematic)
│   └── fig_data.py                   renders the data figures (nonident, rho, VoI, scaling, bound)
└── figures/                          PDF/PNG figures embedded in the manuscript
└── results/
    ├── rho_certificate.json          Phase A output
    ├── voi_runs.csv / voi_summary.json   Phase B sweep
    ├── voi_tau_comparison.json       Horizon corroboration
    ├── scaling.csv / scaling.json    Phase C scaling
    └── voi_scaled.csv / voi_scaled_summary.json   VoI at scale
```

The manuscript with all extension additions marked in red is
`../Lockers_Extension_Q1.tex` (base text black; additions wrapped in `\new{}`).

## Phase A result, in one line

Under the base model's distance-only objective, the **return-flow ratio ρ is not
identified**: over the optimal set it ranges across an interval, and the value
reported in the base paper is an endpoint of that interval — the *max*-ρ endpoint
for the 5-locker instance and the *min*-ρ endpoint for the 15-locker case study,
under identical solver settings. ρ is therefore a solver tie-breaking artifact,
not a model output, and cross-instance/scenario ρ comparisons are unsupported.

| Instance | delivery interval | ρ identified set | Δρ (≥) | paper reports |
|---|---|---|---|---|
| Small, §5 (5 lockers)  | [181, 191] | [0.5201, 0.5452] | 0.0251 | min delivery / **max ρ** |
| Kandoo, §7 (15 lockers) | [425, 435] | [0.3419, 0.3599] | 0.0180 | max delivery / **min ρ** |

The certificate needs **no distance matrix** (routing is fixed to the routes the
paper publishes), so it reproduces entirely from the manuscript — which matters,
since the distance matrices are not published. As a by-product it reproduces the
paper's own reported delivery vectors exactly, at one endpoint of each interval.

## Reproduce

Requires Python 3.9+ and PuLP (bundled CBC solver):

```bash
pip install pulp
python code/certify_rho.py
```

The strong exact baseline uses HiGHS (`pip install highspy`). The real-geography
benchmark additionally needs OSMnx (`pip install osmnx`) and live network access to
OpenStreetMap; it pulls real parcel-locker coordinates and real driving distances,
then checks that the return-flow ratio is still non-identified and the
value-of-integration regime still holds on real street networks:

```bash
pip install osmnx highspy
python code/real_bench_run.py
```

Prints the table above and writes `results/rho_certificate.json`. Runtime is a
few seconds; the flow subproblems are small LPs.

## Method

For each instance the routing is fixed to the paper's published optimal routes.
The flow subproblem (constraints 5–11 of the manuscript, in variables `y`, `z`)
is then solved twice: minimising and maximising total returned flow `R`. Because
the objective of the base model ignores `y` and `z`, both extremes are optimal
for the full model, so the gap between them is a lower bound on the
identification width of ρ. See `theory/rho_nonidentifiability.md` for the formal
statement and proof.

## Data provenance

All instance data (`code/certify_rho.py`) are transcribed verbatim from
`Versión0.tex`: small instance from Section 5 (demand Tables, depot table,
parameters Q=60, H=40); Kandoo case study from Section 7 (demand Tables for both
periods, depot `g`, parameters Q=125, H=30, and the two published daily routes).
