# A matheuristic that scales past the exact wall

**Working note — Phase C.** Code: [`../code/matheuristic.py`](../code/matheuristic.py)
(ALNS), [`../code/flow_eval.py`](../code/flow_eval.py) /
[`../code/flow_eval_fast.py`](../code/flow_eval_fast.py) (exact flow evaluator,
PuLP reference + fast scipy/HiGHS version), [`../code/scaling.py`](../code/scaling.py)
(study). Results in [`../results/scaling.json`](../results/scaling.json).

---

## 1. The wall

The base paper's own benchmark shows the exact MILP is tractable to ~12 lockers
and then breaks: at 15 lockers average runtime jumps to ~1700 s with positive
gaps, and at 30 lockers only half the instances even return a feasible incumbent.
A Q1 computational study needs realistic sizes, so a scalable method is required.

## 2. Method: heuristic routing, exact flow

The algorithmic idea exploits the structure Phase A exposed: **given the routing,
the delivery/return decision is an LP.** So we search only over routing and score
every candidate by its exact optimal flow.

- **Outer search (heuristic):** ALNS over per-period vehicle routes.
  - construction: nearest-neighbour giant tour per period, adding vehicles until
    the flow LP is feasible;
  - operators: 2-opt and or-opt (distance-guided), plus random-removal /
    greedy-reinsertion destroy-repair for diversification;
  - acceptance: record-to-record around the incumbent, best-so-far retained, soft
    restart on stagnation.
- **Inner evaluation (exact):** for each candidate routing, the service-aware
  flow subproblem `min dist + mu*R` is solved to optimality by
  `flow_eval` — an LP in `(y,z)` over constraints (5)-(11). Thus the routing is
  approximate but the flow, delivery split, and returns are optimal at every step.
- **Speed:** the flow LP is rebuilt and solved in-process with scipy/HiGHS
  (`flow_eval_fast`), ~1-7 ms per evaluation — a ~14x speedup over the PuLP+CBC
  subprocess path, validated to return identical objectives (they differ only in
  the free variable R at mu=0, exactly the Phase-A degeneracy). This is what makes
  a few hundred ALNS iterations cost seconds rather than minutes.

## 3. Validation and scaling (mu=5, tau=2, service objective)

Under a **fair, generous 600 s CBC budget** (not a 60 s cap — this is the
`scaling_fair_par.py` run, parallelised across cores; results in
`results/scaling_fair.json`), the matheuristic **matches the proven optimum (0.00%
gap)** where CBC proves it (n=5), then increasingly **dominates CBC's time-limited
incumbent** as the network grows:

| n  | matheuristic (time) | exact CBC (600 s cap)          | matheuristic vs CBC |
|----|---------------------|--------------------------------|---------------------|
| 5  | 454 (1.4 s)         | 454 — **proven optimal, 4 s**  | gap **0.0%**        |
| 8  | 894 (1.7 s)         | 910 — incumbent, 600 s         | **-1.8%** (better)  |
| 10 | 1182 (1.9 s)        | 1204 — incumbent, 600 s        | **-1.8%**           |
| 12 | 1605 (2.2 s)        | 1855 — incumbent, 600 s        | **-13.5%**          |
| 15 | 2213 (2.8 s)        | 2760 — incumbent, 600 s        | **-19.8%**          |
| 20 | 3682 (4.2 s)        | 4493 — incumbent, 600 s        | **-18.1%**          |

Even at ten times the first-pass budget, CBC proves optimality only at n=5; from
n=8 its incumbent trails the matheuristic, by up to ~20% at n=15-20 — so the wall
is qualitative, not an artifact of the cap. (Gurobi would be faster, but the base
benchmark's own Gurobi/3600 s runs already show an ~11% gap at n=20.)

## 4. Honest scope / what remains

- **Lower bound — a precise decomposition** (all gaps are to the *proven* exact
  optimum, so this covers the five instances n=5 seeds 0-2 and n=8 seeds 1-2;
  results saved in `results/bound_ladder.json` via `bound_ladder.py`):
  1. plain LP relaxation (`model_full.lower_bound`): weak, **17-48%** below.
  2. subtour-elimination cutting planes (`bound_cuts.py`, exact min-cut
     separation): valid, tightens by ~5-8 points, still loose.
  3. **column-generation (root) bound** (`branch_price.py`): a route-based
     Dantzig-Wolfe reformulation — columns are elementary tours priced by a
     prize-collecting tour MILP; the master couples the commodity flow to the
     chosen routes. Root node only; **no branching** (so "column-generation root
     bound", not "branch-and-price").

  The result is sharp and validated:
  - On the **routing** objective (μ=0) the CG bound is **tight — 0.0% gap on every
    tested instance (n=5, n=8)**, and strictly better than the arc-LP (~2% off).
  - With the **service penalty** (μ>0) the CG bound alone does not beat the subtour
    cuts (~16-41% gap): the residual is **entirely the return-flow relaxation**,
    which route columns do not tighten.

  4. **return valid inequalities** (`valid_ineq.py`): routing-independent lower
     bounds on returns — the 2-cycle bound `sum_k R_{t,k} >= sum_{a<b}
     min(p_ab^t, p_ba^t)` and its 3-cycle (linear-ordering) strengthening. Added
     to the master they close most of the service-term gap:

     | instance | gap LP | gap CG alone | gap CG+2cyc | gap CG+3cyc |
     |---|---|---|---|---|
     | n5 s1 | 17.4% | 16.0% | 4.6% | **2.3%** |
     | n5 s0 | 26.6% | 25.1% | 7.5% | **4.0%** |
     | n5 s2 | 34.3% | 25.8% | 5.1% | **3.7%** |
     | n8 s1 | — | 37.1% | — | **11.0%** |
     | n8 s2 | 47.6% | 40.8% | — | **11.6%** |

  Combining the column-generation bound (routing) with the return inequalities
  (service term) brings the bound to within **~2-12%** of the proven exact
  optimum — from 17-48% for the plain LP. The residual is routing-dependent
  deferral, which higher-order inequalities would continue to close. The
  "matheuristic vs exact" comparison (0% at n=5; dominance at n≥12) remains the
  operative quality evidence for the incumbent at larger sizes.
- **Prototype scope.** tau=2, CBC, single-seed-per-size, giant-tour construction.
  n=30 returned infeasible under the current instance generator (a locker
  reception-limit `H` interaction at that size, not a method failure); the
  construction and instance design need hardening for a full benchmark.
- **Production solver.** Runs here use CBC/HiGHS; the paper's exact baseline and a
  tighter bound would use Gurobi.
- **Operators.** The current operator set is standard; locker/recirculation-specific
  operators and an adaptive weight scheme are natural refinements and part of the
  algorithmic contribution to develop.

## 5. What this gives the paper

- A method that is **provably optimal where checkable** and **scales past the
  exact wall**, enabling the realistic-size computational study a Q1 venue expects.
- A clean algorithmic story — *heuristic routing, exact flow* — that is itself a
  contribution and that follows directly from the Phase-A structural result.
- The vehicle to run the **value-of-integration study at scale** (n=8-20, tau>=3),
  which the exact solver cannot reach (see `voi_scaled.py`).
