# Why the route-based lower bound does not extend cheaply past n≈15

**Negative result, documented.** We tried the two standard techniques for making
set-partitioning column-generation pricing scale past exact enumeration --- the
**ng-route relaxation** (Baldacci--Mingozzi--Roberti) and **decremental state-space
relaxation (DSSR)** --- and both collapse on this model. The exact bitmask
(Held--Karp) pricer therefore remains the tool, and its practical ceiling of about
15 lockers is a property of the formulation, not of the implementation.

## The pricing subproblem here has no route resource

In the Dantzig--Wolfe reformulation (`branch_price.py`), vehicle capacity is dualized
into the master's arc-linking constraints, so the per-period pricing problem is a
pure prize-collecting elementary tour with reduced arc cost

```
d'(i,j) = d(i,j) + Q · σ(t,i,j),     σ ≤ 0  (capacity dual).
```

Because σ can be strongly negative, `d'(i,j)` is often large and negative (order
−10³ to −10⁴ in our runs). Elementarity (each node at most once, ≤ n arcs) is the
*only* thing that bounds the tour cost from below: there is no time or load resource
on the route to do it.

## Consequence: every state-space relaxation degenerates

- **ng-route.** Forgetting nodes outside a fixed ng-neighborhood lets the walk
  re-collect the strongly-negative arcs in a cycle, so the minimum reduced cost is
  hugely negative and the Lagrangian bound is valid but useless. Measured
  (`results/bound_ng.json`, μ=5): while the exact DP bound is well above the LP and
  below the heuristic upper bound (e.g. n=15: LP 583, **CG 1496**, UB 2207), the
  ng bound is −1.6·10⁵ at n=15 and worsens with n. Verified consistent as a
  relaxation: at ng-size = n it reproduces the exact DP bound exactly.
- **DSSR.** Adding only the nodes that actually cycle to a "must-be-elementary" set
  S gives the *exact* elementary reduced cost when it terminates, and it matches the
  bitmask DP to the digit (n=10 and n=15: identical reduced cost). But the negative
  arcs make **almost every node cycle**, so S grows to nearly n (|S| = 9/10 at n=10,
  15/15 at n=15, 19/20 at n=20). With n·2^{|S|} ≈ n·2ⁿ states the relaxation offers
  no saving over the plain 2ⁿ DP and is in fact slower in Python (n=15: 19 s vs 1 s).

## Takeaway

The obstacle is structural: dualized capacity, with no compensating route resource,
makes the pricing an elementary shortest path riddled with negative cycles, and all
known relaxations of elementarity reintroduce those cycles. Extending the certified
bound past ~15 lockers would need a genuinely different route resource (e.g. a load
or hop bound priced on the route itself) or a different bounding scheme
(Lagrangian/SDP on the arc formulation), not a smarter labelling. Both pricers live
in `branch_price.py` (`_price_ng`, `_price_dssr`) with this caveat, and the
comparison is reproduced by `code/bound_ng.py`.
