# Identifiability of operational indicators in routing-with-recourse models

**Working note — the general result behind C1.** Elevates the specific
non-identifiability of the return-flow ratio (`rho_nonidentifiability.md`) to a
general, checkable principle for a class of optimization models. Journal-agnostic;
the condensed version is integrated in red into the manuscript. Honest positioning:
the underlying phenomenon is the well-known *multiplicity of optimal solutions*
(alternative optima / degeneracy); the contribution is to frame it as an
**identifiability** question for the *operational indicators* that such models
report, to give a checkable characterization and identified set, and to state the
minimal objective repair — in the routing-with-recourse setting.

---

## 1. The class of models

Consider an optimization model whose variables split into two groups,

$$
\min_{x,\,f}\; c(x)\qquad\text{s.t.}\qquad (x,f)\in P,
$$

where

- $x$ are the **primary** (design) variables that the objective depends on --- in
  routing problems, the routing/first-stage decisions;
- $f$ are the **recourse** (operational) variables --- flows, deliveries,
  assignments, returns --- constrained through $P$ but **absent from the
  objective**;
- $P$ is the feasible set (a polyhedron in the LP/flow part for fixed $x$).

This structure is pervasive: routing + commodity flow (the locker model),
facility location + customer assignment, network design + multicommodity routing,
lot-sizing + distribution. In all of them, cost is charged on the design $x$, while
the operational layer $f$ is free to move within feasibility once $x$ is fixed.

Let $z^\star=\min\{c(x):(x,f)\in P\}$, let

$$
X^\star=\{x:\exists f,\ (x,f)\in P,\ c(x)=z^\star\}
$$

be the set of **cost-optimal designs**, let $F(x)=\{f:(x,f)\in P\}$ be the
**recourse polytope** at $x$, and let the **optimal set** be

$$
\mathcal S^\star=\{(x,f):x\in X^\star,\ f\in F(x)\}.
$$

An **operational indicator** is any function $g:P\to\mathbb R$ that a study reports
--- an average utilization, a return ratio, a same-day-service rate, a load factor.
Typically $g$ depends on the recourse $f$.

## 2. Identifiability

> **Definition.** The indicator $g$ is **identified** by the model if it takes the
> same value at every optimal solution, i.e. $g$ is constant on $\mathcal S^\star$.
> Otherwise its **identified set** is $g(\mathcal S^\star)=\{g(s):s\in\mathcal S^\star\}$.

The point of the definition: a single reported value of $g$ is a property of *the
model* only if $g$ is identified. If not, the value returned by a solver is a
property of the solver's tie-breaking, not of the system, and comparisons of it
across instances, sizes, or scenarios conflate the two.

> **Proposition 1 (characterization and identified set).**
> $g$ is identified iff (i) for every $x\in X^\star$, $g(x,\cdot)$ is constant on
> $F(x)$, and (ii) that constant does not depend on $x\in X^\star$. If $g$ is
> linear in $f$ and each $F(x)$ is a polytope, then for a fixed $x\in X^\star$ the
> restriction $g(\{x\}\times F(x))$ is the interval $[\underline g(x),\overline
> g(x)]$ whose endpoints are the two linear programs $\min/\max\{g(x,f):f\in
> F(x)\}$; and $g(\mathcal S^\star)\supseteq[\underline g(x),\overline g(x)]$.

*Proof.* Immediate from the definition: $g$ constant on $\mathcal S^\star$ iff
constant on each fibre $\{x\}\times F(x)$ and equal across fibres. For fixed $x$,
$F(x)$ is a polytope (nonempty, since $x\in X^\star$) and $g(x,\cdot)$ linear, so
its image is the interval between the two LP optima; the union over $x\in X^\star$
contains any single fibre's interval. $\square$

Proposition 1 makes identifiability **checkable**: fix any one optimal design
$x^\star$ (e.g. one returned by a solver) and solve two LPs. If they disagree, $g$
is not identified and the gap is a *lower bound* on the width of the identified set
(the true set, over all optimal designs, can only be wider).

> **Proposition 1b (the identified set over the cost-optimal set).**
> When the design variables $x$ are (partly) integer — as the routing variables of a
> VRP are — the cost-optimal designs $X^\star$ do **not** form the optimal face of a
> single LP; they are a mixed-integer set, and
> $g(\mathcal S^\star)=\{g(x,f):x\in X^\star,\ f\in F(x)\}=\bigcup_{x\in X^\star}I(x)$
> is the image of $g$ over a *union* of recourse polytopes, where $I(x)=g(\{x\}\times
> F(x))$. (i) The two-LP interval $I(x^\star)$ of Proposition 1 at a fixed $x^\star$ is
> contained in $g(\mathcal S^\star)$, with equality iff no other optimal design
> enlarges the range, i.e. $I(x)\subseteq I(x^\star)$ for all $x\in X^\star$
> (invariance of $I(\cdot)$ is a sufficient special case); in general a conservative
> inner bound. (ii) Since $X^\star$ is discrete, $g(\mathcal S^\star)$ may be a union
> of disjoint intervals; its extreme values (range) solve $\min/\max\{g(x,f):(x,f)\in
> P,\ x\ \text{integer},\ c(x)=z^\star\}$ — a *mixed-integer* program — exact as an
> interval when connected, which already decides identification. Computing it needs
> $z^\star$ (NP-hard for the routing class) and then an optimization of $g$ over the
> cost-optimal set, whereas the fixed-design test is two LPs.

*Proof.* $\mathcal S^\star=\bigcup_{x^\star\in X^\star}\{x^\star\}\times F(x^\star)$,
so $g(\mathcal S^\star)=\bigcup_x I(x)$. (i) $I(x^\star)$ is $g$ over one fibre, hence
a subset; the union equals it iff every other $I(x)\subseteq I(x^\star)$. (ii) The
extreme values optimize $g$ over $\{x\ \text{integer},\ c(x)=z^\star\}$; a MILP, and
fixing $x=x^\star$ drops integrality to the two LPs. Holds for any $g$ with interval
fixed-design image, incl. quasi-linear $\rho$ (Charnes–Cooper). $\square$

This is the honest separation behind the framing: the *fixed-design* half is the
classical alternative-optima fact (two LPs, polynomial); the *identified set over the
mixed-integer cost-optimal set* is the new, hard object, for which the two LPs are
only a certified inner bound. **Instantiated** (`code/optface_rho.py`,
`results/optface_rho.json`): on 12 five-locker instances the exact range over the
cost-optimal set is strictly wider than the single-routing inner bound in 10, e.g.
inner $[0.45,0.58]$ (width 0.14) inside exact $[0.21,0.58]$ (width 0.37, 2.7×). So the
locker certificate is a genuine inner bound and understates $\rho$'s non-identification.

> **Proposition 2 (sufficient condition via free recourse).**
> Suppose the objective is a function of $x$ alone. If there exist $x^\star\in
> X^\star$ and $f_1,f_2\in F(x^\star)$ with $g(x^\star,f_1)\ne g(x^\star,f_2)$,
> then $g$ is not identified. In particular, if $g$ is linear in $f$ and $F(x^\star)$
> has a feasible direction $d$ (an edge or a lineality direction) along which the
> recourse can move and $\nabla_f g\cdot d\ne 0$, then $g$ is not identified.

*Proof.* $c$ depends only on $x$, so every $(x^\star,f)$ with $f\in F(x^\star)$ has
objective $c(x^\star)=z^\star$ and lies in $\mathcal S^\star$; hence $f_1,f_2$ give
two optimal solutions with different $g$. The directional statement is the special
case $f_2=f_1+\epsilon d$. $\square$

The condition is exactly "the objective does not price the recourse that $g$
measures." Whenever the operational layer has slack that the cost ignores, any
indicator reading that slack is unidentified.

## 3. Repairing identifiability

There are two ways to make $g$ identified, and they are qualitatively different.

> **Proposition 3 (lexicographic identification --- no cost change).**
> Let $h:P\to\mathbb R$ be a recourse objective such that, for each $x\in X^\star$,
> $\arg\min\{h(x,f):f\in F(x)\}$ has a unique $g$-value $g^\ast(x)$. Solving the
> lexicographic problem "$\min c(x)$, then $\min h$" leaves $X^\star$ and the cost
> $z^\star$ unchanged and identifies $g$ to $g^\ast(x)$ on each fibre. If moreover
> $g^\ast(x)$ is constant across $X^\star$, $g$ is fully identified at no cost.

*Proof.* The lexicographic tie-break selects, among cost-optimal solutions, those
minimizing $h$; by hypothesis these share the value $g^\ast(x)$. Cost is untouched
because $h$ breaks ties only within $\mathcal S^\star$. $\square$

> **Proposition 4 (penalized identification --- $g$ becomes a decision).**
> For $\mu>0$, the reformulated model $\min\, c(x)+\mu\, h(f)$ makes the recourse
> $h$ (hence any $g$ pinned by $h$) part of the objective. For $\mu$ small enough to
> preserve $X^\star$ this coincides with Proposition 3; for finite $\mu$ it trades
> design cost against the operational term, turning $g$ from an unidentified
> by-product into a genuine, tunable operating point.

*Proof.* Standard: adding $\mu h$ to the objective prices the recourse; the optimal
recourse now optimizes $c+\mu h$ rather than being free, so any $g$ determined by
$h$ is determined by the model. The $\mu\to0^+$ limit recovers the lexicographic
selection. $\square$

The distinction matters for practice: Proposition 3 says an operator can make a
reported indicator meaningful **without changing costs** (a reporting convention);
Proposition 4 says that if the indicator reflects a real service objective, pricing
it changes the plan and yields a routing/service trade-off curve.

## 4. The locker model as a corollary

In the multi-period locker VRP, $x$ is the routing $x_{ij}^t$, the recourse $f$ is
the commodity flow/delivery/return $(z,y)$, and the objective (1) is $\sum_t\sum_{ij}
d_{ij}x_{ij}^t$ --- a function of $x$ alone. Constraint (11) prices only the
depot-dispatched flow; a locker-origin parcel whose destination is later on the
route may be delivered or returned within $F(x)$ at no cost. The return-flow ratio

$$
\rho=\frac{R}{Y+R},\qquad R=\text{total returns},\quad Y=\text{total deliveries},
$$

is a linear-fractional function of the recourse. By Proposition 2 it is **not
identified**; by Proposition 1 its identified set is the interval computed by the
two flow LPs of the certificate (small instance $[0.520,0.545]$, case study
$[0.342,0.360]$). The service-aware objective $\sum d\,x+\mu R$ is exactly the
Proposition-4 repair with $h=R$: it makes $\rho$ a decision and traces the
routing/timing trade-off (the $\rho(\mu)$ curve). Penalizing returns is also the
Proposition-3 lexicographic repair in the $\mu\to0^+$ limit, which would identify
$\rho$ at its minimum-recirculation value without changing routing cost.

## 5. Scope and honesty

- The underlying *fixed-design* fact --- optimal solutions need not be unique, and a
  solver returns one arbitrarily --- is classical (alternative optima, degeneracy).
  The contribution is **not** that fact but: the **identifiability lens on reported
  operational indicators**; the **separation of the tractable part (fixed design: two
  LPs) from the hard part (the identified set over the mixed-integer optimal face, a
  MILP)** (Prop. 1b); the **two-LP checkable test / certified inner bound**; and the
  **lexicographic-vs-penalized repair dichotomy** — stated for the routing-with-recourse
  class and instantiated on a real model. It is a framing + structure/complexity
  contribution, not new polyhedral theory.
- Proposition 1's identified set is exact for a fixed optimal design and a lower
  bound over all optimal designs (computing the exact set over $X^\star$ is as hard
  as optimizing $g$ subject to $c=z^\star$).
- The principle is a reporting discipline: **before reporting or comparing an
  operational indicator from such a model, test whether the objective prices it; if
  not, report the identified interval or repair the objective.**
