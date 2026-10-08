# 09 — Neighbors and separating surfaces

**Prerequisites:** distances and dot products (03), evaluation (06), and binary labels. **Objective:** explain how geometry changes a prediction, inspect a margin objective, and understand what a kernel computes.

Nearest neighbors stores training examples and finds the `k` closest to a new query. Here `X` has shape `(n,p)`, `y` has shape `(n,)`, and the query has shape `(p,)`. Euclidean distance is `sqrt(sum((x-query)²))`. For binary labels, averaging selected labels estimates a local class-one fraction. Weighted averaging uses inverse distance. These fractions describe the chosen neighborhood; they are not automatically calibrated population probabilities.

Consider positions `[0,1,2,6,8]` with labels `[0,0,0,1,1]`. At query 4.4 the nearest position is 6, so one-neighbor classification returns class one. Three neighbors are 6, 2, and 1, with distances 1.6, 2.4, and 3.4. Two have label zero, so the unweighted probability becomes 1/3. A change to `k` changes the effective smoothing assumption.

```bash
python -m mlfirst --lesson 09
```

Inspect `probability_k1`, `probability_k3`, and `probability_weighted_k3`. Inverse-distance weighting raises the class-one probability to about 0.468 but does not quite cross 0.5. `neighborhood_radius_k3` is 3.4. The radius exposes how far the method must reach for evidence. Identical fractions can arise in well-sampled and nearly empty regions.

The implementation resolves equal-distance ties by original input order. In weighted mode, selected exact matches are averaged and nonzero-distance neighbors ignored. If more duplicates exist than `k`, only selected duplicates contribute. This explicit convention prevents division by zero while leaving label conflicts visible to the analyst.

A separating surface uses score `w.T@x+b`. With labels encoded -1/+1, the signed margin is `y*score`. Our educational SVM-style objective is `mean(max(0,1-y*score)) + λ||w||²/2`. It is different in normalization from a summed hinge objective. `hinge_loss_gradient` exposes loss and subgradients; it is not a fitted SVM solver.

For a negative example at 0 and positive example at 2, choose `w=1,b=-1`. Both signed margins are one and hinge loss is zero. `margin_width=2` because the two unit-score planes are `2/||w||` apart. `regularized_hinge_objective=0.5` includes the weight penalty. The reported subgradient chooses zero for the hinge contribution at its corner; other valid subgradients exist there.

An RBF kernel computes `exp(-gamma*||x-y||²)`. The demo's Gram matrix has shape `(5,5)` and a nonnegative `rbf_smallest_eigenvalue` up to rounding. This finite check illustrates, rather than proves, the kernel's positive-semidefinite property. Standardization can improve geometry, but fit its statistics on training data. Irrelevant dimensions, arbitrary units, and distant queries remain substantive failure modes.

**Practice**

1. Derive the weighted probability above from three reciprocals.
2. Double all distances but retain their labels. What changes under inverse-distance weighting?
3. Compute the RBF similarity at distance 2 with gamma 0.5.

<details><summary>Worked solutions</summary>

1. `(1/1.6)/(1/1.6+1/2.4+1/3.4)=51/109≈0.468`.
2. Every weight halves; their normalized proportions stay unchanged. The neighborhood radius doubles, signaling weaker local support in those units.
3. `exp(-0.5×4)=exp(-2)≈0.1353`. Larger gamma reduces similarity at any fixed nonzero distance.

</details>

**Scope and source:** brute-force binary kNN, an SVM objective, and RBF similarities are supplied. Kernel SVM optimization, search indexes, metric learning, and calibration are not. See [implementation](../mlfirst/classical.py), [book companion](../book_code/companion_classical.py), and [checks](../tests/test_classical.py).
