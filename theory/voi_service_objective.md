# Service-aware objective and the value of multi-period integration

**Working note — Phase B.** Builds on the Phase-A result
([`rho_nonidentifiability.md`](rho_nonidentifiability.md)). Code:
[`../code/model_full.py`](../code/model_full.py) (full MILP, constraints 2–14 of
`Versión0.tex`) and [`../code/voi.py`](../code/voi.py) (experiment). Results in
[`../results/voi_summary.json`](../results/voi_summary.json) and
`voi_runs.csv`. Journal-agnostic; the manuscript-facing pieces are integrated in
red into `Lockers_Extension_Q1.tex`.

---

## 1. The reformulated objective

Phase A showed that under the base objective `min Σ_t Σ_(i,j) d_ij x_ij^t` the
same-day delivery total — and hence the return-flow ratio ρ — is *not a
decision*: it is free over the optimal set. To make same-day completion a
modelled quantity we add a deferral penalty:

```
min  Σ_t Σ_(i,j) d_ij x_ij^t  +  μ · Σ_t Σ_k Σ_{i∈δ⁻(0)} z_i0k^t
                                   └──────── R = total returned flow ────────┘
```

- `μ = 0` recovers the base model.
- `μ > 0` makes ρ identifiable (the degeneracy of Phase A is removed).
- Linear, adds no variables. This is the paper's own suggested direction
  (Conclusions, "penalize return flows to the depot or the postponement").

Bi-objective and same-day-reward (`−λ Σ y`) variants are equivalent; the
deferral-penalty form is adopted because it acts directly on the recirculation
mechanism that defines the multi-period model.

## 2. Value of integration: definition

Two policies for the τ-period horizon:

- **JOINT** — optimize the whole horizon at once (the paper's model).
- **MYOPIC** — rolling horizon: solve each day in sequence, carrying returns
  forward, with no lookahead. (This is the paper's own suggested benchmark,
  "day-by-day optimization without inter-period recirculation".)

Under one global objective `obj = distance + μ·R`, any myopic solution is
feasible for the joint problem, so

```
VoI = obj_myopic − obj_joint ≥ 0          (theoretical guarantee)
```

VoI is the rolling-horizon suboptimality — the price of not integrating. The
empirical question is **when** it is large. We sweep capacity tightness × μ over
controlled synthetic instances (n=5, τ=2, 12 seeds; Euclidean distances). Joint
solves are capped at 25 s, so a non-zero VoI is a *conservative lower bound*
(CBC's incumbent over-estimates obj_joint).

## 3. Result: integration is a tight-capacity phenomenon

| Regime | μ | VoI>0 | mean VoI% | max VoI% |
|---|---|---|---|---|
| loose (Q=60) | 0 | 0/12 | 0.00 | 0.00 |
| loose (Q=60) | 5 | 0/12 | 0.00 | 0.00 |
| tight (Q=22) | 0 | 5/12 | 1.76 | 7.85 |
| tight (Q=22) | 5 | 4/12 | 1.61 | **10.58** |

**Reading:**

1. **Under loose capacity, integration is exactly inert** — VoI = 0 in all 24
   runs, both objectives. Day-by-day optimization loses nothing: there is no
   cross-period coupling to exploit when capacity never binds.
2. **Tight capacity activates integration.** The coupling is the recirculation
   load: day-1 deferrals become day-2 depot dispatch that must be delivered
   (constraint 11), consuming day-2 capacity. A myopic day-1 that greedily
   minimizes its own cost can overload day 2; the joint plan trades a little
   day-1 cost to relieve it. VoI reaches ~8–11% of objective.
3. **Both channels are real.** At μ=0 the gap is in *distance* (capacity coupling
   alone). At μ>0 the *service* channel adds the largest single gaps (10.58%).

This matches, and sharpens, the earlier proof-of-concept: not "joint always
wins", but "joint wins exactly when capacity is binding". Characterizing the
*regime* (rather than asserting a uniform dominance) is the contribution that
carries operational weight.

## 4. Honest scope / what remains

- **Small and short.** n=5, τ=2, CBC. The regime is clear, and the coupling
  **strengthens sharply with the horizon**: under the same tight regime and
  service objective (μ=5, seeds 0–7), VoI is positive in 2/8 instances at τ=2 but
  **7/8 at τ=3** (mean 0.52%→2.48%, max 3.75%→6.14%; `voi_tau_comparison.json`).
  The value of integration compounds over periods — consistent with the earlier
  proof-of-concept. Scaling to n=8–20, τ≥3 with Gurobi, plus a proper statistical
  design (regression of VoI on tightness/μ/demand), is the next step and the
  basis of the paper's computational study.
- **Conservative VoI.** Joint capped at 25 s → reported VoI ≤ true VoI.
- **Tightness knob.** "tight/loose" here are two Q values; the paper version
  should sweep a continuum and locate the binding threshold precisely.
- **Not yet a method.** JOINT is solved exactly, which does not scale past
  ~12–15 lockers (base paper's own tractability wall). Turning this into a
  large-scale study needs the scalable matheuristic (Phase C).

## 4b. Value of integration at scale (via the matheuristic)

Using the Phase-C matheuristic (`voi_scaled.py`, n=8–20, τ=3, tight-feasible
capacity, μ=5, 6 seeds each) lets us probe sizes the exact solver cannot reach.
The joint search is **warm-started from the myopic solution** (which is feasible
for the joint problem), so the joint objective starts at `obj_myopic` and only
improves. `VoI = obj_myopic − obj_joint` is then a *genuine, non-clamped* measure
of the cross-period coordination the joint plan finds — not an artifact of two
independent heuristic runs.

| n | frac VoI>0 | mean VoI% | max VoI% |
|---|---|---|---|
| 8  | **6/6** | 2.57 | **8.62** |
| 10 | 2/6 | 0.32 | 1.64 |
| 12 | 4/6 | 1.32 | 3.77 |
| 15 | 3/6 | 0.28 | 1.25 |
| 20 | 2/6 | 0.27 | 1.14 |

**Reading.** With the correct (warm-started) measurement, the value of integration
is both **prevalent and material** at scale: it is positive in 33–100% of
tight-capacity instances (averaging ~57% across sizes), reaching **8.6%** of the
objective. Prevalence tracks how hard capacity binds — n=8 (tightest relative to
its per-vehicle need) shows it in every instance — which is exactly the
tight-capacity regime characterization of Phase B, now confirmed at scale. The
earlier under-measurement (∼1/6) came from running the joint and myopic searches
independently; warm-starting closed that gap. Value of integration is not
universal, but it is a real, recurring, capacity-driven effect, not heuristic
noise.

**A tightness knob, honestly.** These are the tight-feasible band (Q ≈ 0.5× the
reference flow); loosen capacity and the value of integration falls to zero
(Phase B). A continuous sweep locating the binding threshold per size is the
clean next experiment.

## 5. What this gives the paper

- A **modelling fix** (service-aware objective) that makes ρ and the operational
  indicators meaningful — closing the gap Phase A opened.
- A **value-of-integration analysis** with a *regime characterization*: the
  multi-period model earns its complexity precisely under tight capacity. This is
  the empirical core that distinguishes the extension from the base paper, whose
  own multi-period mechanism is (Phase A) operationally inert under its stated
  objective.
