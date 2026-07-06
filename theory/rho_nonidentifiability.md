# Non-identifiability of the return-flow ratio under the distance-only objective

**Working note — Phase A.** Result and proof in the notation of the base
manuscript (`Versión0.tex`). Journal-agnostic. Intended to be adapted, in red,
into a subsection of the extended paper. The computational certificate backing
Proposition 2 is `../code/certify_rho.py` (output in
`../results/rho_certificate.json`).

---

## 1. Setup and notation

We keep the base model exactly as formulated in the manuscript. Let

$$
\mathcal{M}:\quad
\min \; \sum_{t\in T}\sum_{(i,j)\in E} d_{ij}\,x_{ij}^{t}
\quad\text{s.t. constraints (2)–(14).}
$$

Denote by $\mathcal{P}$ the feasible region in the variables $(x,y,z,w)$, by
$Z^\star$ the optimal objective value, and by

$$
\mathcal{S}^\star \;=\; \bigl\{ (x,y,z,w)\in\mathcal{P} \;:\; \textstyle\sum_{t}\sum_{(i,j)} d_{ij}\,x_{ij}^{t}=Z^\star \bigr\}
$$

the set of optimal solutions. The paper defines the **return-flow ratio**

$$
\rho(x,y,z,w) \;=\;
\frac{R}{\,Y+R\,},\qquad
R=\sum_{t\in T}\sum_{k\in K}\sum_{i\in\delta^-(0)} z_{i0k}^{t},
\qquad
Y=\sum_{t\in T}\sum_{k\in K} y_k^{t},
$$

and reports and analyses $\rho$ as an operational indicator throughout the
benchmark (Table on operational indicators by size and by scenario) and the case
study (summary-indicators table). Here $R$ is total returned flow and $Y$ total
delivered flow.

The question this note settles is **whether $\mathcal{M}$ determines $\rho$** —
i.e. whether $\rho$ is constant across the optimal set $\mathcal{S}^\star$, or
only known up to a range.

---

## 2. Objective blindness

> **Lemma 1 (objective blindness).**
> The objective of $\mathcal{M}$ is a function of the routing variables $x$
> alone. Consequently, for any cost-optimal routing $x^\star$, every flow
> $(y,z,w)$ that is feasible with $x=x^\star$ yields a point of
> $\mathcal{S}^\star$. Writing $\mathcal{F}(x^\star)$ for the polytope of flows
> feasible with routing $x^\star$,
> $$
> \mathcal{S}^\star \;=\; \bigcup_{x^\star\in X^\star} \{x^\star\}\times \mathcal{F}(x^\star),
> $$
> where $X^\star$ is the set of cost-minimal routings.

*Proof.* The delivery, commodity-flow and load variables $y,z,w$ do not appear
in $\sum_t\sum_{(i,j)} d_{ij}x_{ij}^t$. Thus the objective value of a feasible
solution is unchanged by replacing $(y,z,w)$ with any other flow feasible for the
same $x$. Fixing $x=x^\star$ with $\sum d\,x^\star=Z^\star$, every
$(y,z,w)\in\mathcal{F}(x^\star)$ gives objective $Z^\star$, hence lies in
$\mathcal{S}^\star$. $\qquad\blacksquare$

The point of Lemma 1 is that $\mathcal{M}$ pins down $\rho$ **only to the extent
that the flow constraints (5)–(11) pin down $R$ over $\mathcal{F}(x^\star)$.**
They do not, and the reason is structural.

> **Remark (the free variable).**
> Along a route, a package of the locker-origin demand $p_{jk}^t$ whose
> destination $k$ is visited *after* $j$ can be delivered on the same day
> (raising $y_k^t$, lowering $R$) **or** carried past its destination and back to
> the depot (lowering $y_k^t$, raising $R$). Constraint (11),
> $y_k^t \ge \sum_{j\in\delta^+(0)} z_{0jk}^t$, forces same-day completion **only
> for depot-dispatched flow**; it imposes no upper structure linking
> locker-origin deliverable flow to $y$. Every such deliverable unit is therefore
> free to be delivered or deferred, within vehicle capacity (7) and the reception
> limit (8), at *identical routing cost*. This free choice is exactly the
> degeneracy that Lemma 1 exposes.

---

## 3. Non-identification

Since $\rho$ is a function on $\mathcal{S}^\star$, the model identifies it only
up to the **identified set** $\rho(\mathcal{S}^\star)=\{\rho(s):s\in\mathcal{S}^\star\}$.
$\rho$ is identified iff this set is a singleton.

> **Definition (identification width).**
> $\Delta_\rho \;=\; \max_{s\in\mathcal{S}^\star}\rho(s)\;-\;\min_{s\in\mathcal{S}^\star}\rho(s).$
> $\rho$ is identified iff $\Delta_\rho=0$.

> **Proposition 2 (non-identifiability).**
> For the manuscript's own instances, $\rho$ is **not** identified by
> $\mathcal{M}$. With the routing fixed to the paper's *published* optimal routes
> (a conservative restriction of $\mathcal{S}^\star$, see §4):
>
> | Instance | delivery interval $[Y_{\min},Y_{\max}]$ | $\rho(\mathcal{S}^\star)\supseteq$ | $\Delta_\rho \ge$ |
> |---|---|---|---|
> | Small, §5 (5 lockers) | $[181,\,191]$ | $[0.5201,\,0.5452]$ | $0.0251$ |
> | Kandoo, §7 (15 lockers) | $[425,\,435]$ | $[0.3419,\,0.3599]$ | $0.0180$ |

*Proof.* Fix the routing $x^\star$ to the published optimal routes of each
instance (Section 5 uses $0\!\to\!3\!\to\!1\!\to\!0$ and
$0\!\to\!5\!\to\!2\!\to\!4\!\to\!0$ in both periods; Section 7 uses the two daily
tours reported in the case study). By Lemma 1, $\{x^\star\}\times\mathcal{F}(x^\star)\subseteq\mathcal{S}^\star$.
Minimising and maximising $R$ (equivalently $-Y$) over $\mathcal{F}(x^\star)$ —
two linear programs in $(y,z)$ subject to (5)–(11) — yields the two distinct
feasible extremes tabulated above (certificate: `certify_rho.py`). Both are
optimal for $\mathcal{M}$ by Lemma 1, and they exhibit different $\rho$. Hence
$\rho(\mathcal{S}^\star)$ is not a singleton and $\Delta_\rho\ge$ the tabulated
values. $\qquad\blacksquare$

> **Proposition 3 (the reported value is an endpoint, and the choice of
> endpoint is unstable).**
> The single-valued $\rho$ reported in the manuscript coincides with an
> **endpoint** of $\rho(\mathcal{S}^\star)$ in each instance, and with **opposite**
> endpoints across the two:
> - Small instance: the reported solution attains $Y_{\min}=181$ — the *least*
>   same-day delivery, i.e. the **largest** $\rho$ of the optimal set.
> - Case study: the reported solution attains $Y_{\max}=435$ — the *most*
>   same-day delivery, i.e. the **smallest** $\rho$ of the optimal set.
>
> Both instances were solved by the same solver under identical default settings.
> A model-determined quantity cannot land on opposite extremes of its own optimal
> face under a fixed rule; therefore the reported $\rho$ reflects branch-and-bound
> tie-breaking, not a property of $\mathcal{M}$.

*Verification.* Each reported solution is reproduced exactly at the endpoint the
paper happens to sit on — the **max-$R$** (min-delivery) extreme for the small
instance, $y^1=(15,12,7,16,2),\,y^2=(28,26,26,17,32)$, and the **min-$R$**
(max-delivery) extreme for Kandoo, matching the full Table of delivered packages
by destination and period. That the reproduced endpoint is the *opposite* one in
each instance is exactly the instability of Proposition 3. The identification is
thus exhibited against the paper's published numbers, not a re-solve. See
`rho_certificate.json`. $\qquad\square$

---

## 4. Scope, honesty, and what it does *not* claim

- **Conservative interval.** The intervals above fix routing to the *single*
  published optimal routing. If $\mathcal{M}$ admits alternative cost-optimal
  routings ($|X^\star|>1$), then $\mathcal{S}^\star\supseteq\{x^\star\}\times\mathcal{F}(x^\star)$
  is strictly larger and $\Delta_\rho$ can only grow. The reported widths are
  therefore **lower bounds** on the true identification width.
- **No distance data needed.** Because routing is fixed to the published routes,
  the certificate uses no distance matrix. It is fully reproducible from the
  manuscript, which is relevant since the distance matrices are not published.
- **Not an error in the solved values.** The paper's reported solutions are
  feasible and optimal; we reproduce them. The claim is about *identification*:
  the objective does not determine $\rho$, so any single reported $\rho$ is one
  admissible value among an interval.
- **What breaks the degeneracy.** Adding a service term to the objective (e.g.
  $\min \sum_t\sum_{(i,j)} d_{ij}x_{ij}^t - \lambda\, Y$, or a penalty on $R$)
  makes $Y$ — and hence $\rho$ — a genuine decision. This is the modelling fix
  that also activates the value of multi-period integration; it is developed in
  the subsequent phases.

---

## 5. Managerial corollary

> **Corollary 4.**
> Any comparison of $\rho$ across sizes, scenarios or instances under
> $\mathcal{M}$ conflates a modelled quantity with solver tie-breaking. In
> particular, reported cross-scenario differences in $\rho$ of order
> $\lesssim \Delta_\rho \approx 0.02$–$0.03$ fall within the per-instance
> identification width and carry no operational meaning under the distance-only
> objective. Claims that "the return-flow ratio increases with network size / with
> tighter capacity" are, as stated, not supported by $\mathcal{M}$: they require
> an objective that determines $\rho$.

This turns a defect of the base model into the entry point of the extension: to
speak about recirculation as an operational outcome, the objective must make
same-day completion a decision. That is precisely the reframing the Q1 extension
builds on.
