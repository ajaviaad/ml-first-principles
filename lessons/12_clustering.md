# 12 — Clustering

**Prerequisites:** Euclidean geometry (03), means, and the distinction between supervised and unsupervised learning (01). **Objective:** execute k-means by hand, inspect its objective trace, and demonstrate dependence on initialization.

Clustering asks for a grouping under an explicit notion of similarity. It does not receive a target label that defines the right answer. K-means represents each group with a center and minimizes `sum_i ||x_i-center[label_i]||²`. Input `X` has shape `(n,p)`, initial centers `(k,p)`, and returned labels `(n,)`. The number of groups `k` is a choice supplied by the analyst, not something the basic algorithm discovers.

Lloyd's algorithm alternates two exact conditional steps: assign each observation to its nearest center, then replace each center with its assigned observations' mean. Each step cannot increase the objective when clusters remain nonempty. That descent does not guarantee a global minimum. The algorithm stops here when center values stop changing exactly, or fails explicitly after the iteration limit. Equal-distance ties choose the first center.

For scalar observations `[0,1,4,9]`, initialize centers at 0 and 9. The first three observations join the first group, and 9 joins the second. Their updated centers are `5/3` and 9. The first group's SSE is `(0-5/3)²+(1-5/3)²+(4-5/3)²=26/3`. The second contributes zero. Those assignments remain unchanged after updating.

```bash
python -m mlfirst --lesson 12
```

`labels` is `[0,0,0,1]`; `centers` is approximately `[1.667,9]`. `objective_trace` records the nearest-center SSE at each assignment step, including the initial value 17 and the converged value `26/3`. The final returned labels correspond to the returned centers because convergence is checked before returning.

Now initialize at 0 and 4. The assignments become `[0,0,1,1]`, yielding centers .5 and 6.5. This is another fixed point: its SSE is .5+12.5=13. Inspect `alternate_labels`, `alternate_centers`, and `alternate_objective_trace`. `preferred_run` selects the lower-SSE run, called `first` here. That selection judges the specified objective, not which grouping has the most useful scientific interpretation. Both runs are completely deterministic; this particular lesson does not use its seed.

The algorithm refuses an empty cluster, since its mean is undefined. Duplicate initial centers can create this situation. A practical learner might restart or choose a replacement center; this transparent implementation asks the caller to make that choice. Features with larger numerical units dominate squared distance. Standardization changes the clustering question and should be justified rather than applied automatically.

K-means favors compact groups in Euclidean space. A curved population, unequal densities, and strong outliers can produce misleading divisions. Cluster identifiers have no intrinsic meaning: swapping center order swaps labels without changing the partition. Validate usefulness with domain evidence, robustness to preprocessing and initialization, and a criterion appropriate to the downstream purpose. A tidy plot or low SSE alone does not establish natural categories.

**Practice**

1. Calculate the SSE for centers .5 and 6.5 without running code.
2. Swap initial centers 0 and 9. Which outputs should change?
3. Add 100 to every observation and initial center. Predict the objective trace.

<details><summary>Worked solutions</summary>

1. Residuals are `[-.5,.5,-2.5,2.5]`, so SSE is `.25+.25+6.25+6.25=13`.
2. Numeric labels and center order swap. Membership, distances and SSE do not change, provided tie handling does not affect assignments.
3. Every distance and residual is unchanged. Centers shift by 100 while the objective trace is identical, apart from floating-point rounding.

</details>

**Scope and source:** explicit-initialization Lloyd k-means is implemented. Hierarchical clustering, DBSCAN, fuzzy memberships, Gaussian mixture EM, and silhouette evaluation discussed in the chapter are not supplied here. See [validated wrapper](../mlfirst/classical.py), [original algorithm](../book_code/companion_classical.py), and [invariant tests](../tests/test_classical.py).
