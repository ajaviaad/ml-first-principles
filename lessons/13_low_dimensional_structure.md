# 13 — Low dimensional structure

**Prerequisites:** matrix multiplication, projections and eigenvectors (03), plus the idea of training-only preprocessing (02). **Objective:** connect PCA to singular value decomposition, quantify lost information, and apply a learned transform to new data.

Principal component analysis preserves directions of greatest variation in centered data. For input `X` shaped `(n,p)`, subtract the training column means to obtain `Xc`. Its singular value decomposition is `Xc=U diag(s) V.T`. The first `r` rows of `V.T` are retained components, shaped `(r,p)`. Scores have shape `(n,r)` and equal `Xc @ components.T`. Reconstruction multiplies scores by components and restores the training means.

The squared reconstruction error is the sum of squares of discarded singular values. Sample variance along component `j` is `s[j]²/(n-1)`. The retained fraction of total centered variance is `sum(s[:r]²)/sum(s²)`. These are geometric guarantees under squared error; they say nothing directly about future prediction or semantic importance.

A two-feature covariance matrix `[[4,3],[3,4]]` has eigenvalues 7 and 1. Its leading direction is `(1,1)/sqrt(2)`: the two features vary together. Keeping it retains `7/(7+1)=87.5%` of variance. The discarded contrast direction `(1,-1)/sqrt(2)` has less variance but could still contain the only signal distinguishing two classes.

```bash
python -m mlfirst --lesson 13
```

The runnable example uses a separate four-row, three-feature matrix. `singular_values` is approximately `[sqrt(10),sqrt(2),0]`. `first_component` points along `[2,0,1]` after normalization, possibly with its sign reversed. `scores` gives each row's coordinate on that axis. `retained_variance_fraction` is `10/12=5/6`, distinct from the separate covariance example's `covariance_example_retained_fraction=7/8`. `reconstruction_sse` and `discarded_singular_energy` both equal 2. The equality is the main numerical check.

Use `pca_fit` once on training rows, then `pca_transform` with that fitted model for validation or test observations. Recomputing a new mean separately for test data would silently change the transformation. `pca_inverse_transform` maps scores back to feature units. The model stores all available singular values and explained variances, while `components` includes only the retained directions. Constant input has zero total variance; this implementation defines every variance ratio as zero rather than dividing by zero.

Centering and standardizing are different choices. Centering removes the location; standardization also changes relative feature scales. A wavelength with large meaningful variation should not necessarily be treated identically to a nearly constant noisy sensor. PCA ignores labels, can be sensitive to outliers, and only models linear subspaces. Individual component signs are arbitrary. Near-equal singular values also allow the corresponding directions to rotate, so compare subspaces rather than demanding identical coordinates from every fit.

**Practice**

1. With singular values `[6,3,1]`, compute rank-two reconstruction SSE and retained variance.
2. Negate one component and its score column. Does reconstruction change?
3. Why is fitting PCA to the full dataset before splitting a leakage risk even though PCA never reads labels?

<details><summary>Worked solutions</summary>

1. The discarded squared value is 1. Total energy is 46, so retained variance is `45/46≈97.83%`.
2. No. The two sign changes cancel in the matrix product. The represented axis and reconstruction are identical.
3. Test observations influence the learned mean and directions. Evaluation then uses a preprocessing model that already saw the test distribution, instead of measuring the intended training-only procedure.

</details>

**Scope and source:** centered SVD-based PCA and reusable transforms are implemented. Linear discriminant analysis, t-SNE, UMAP, sparse truncated SVD, and latent Dirichlet allocation are distinct chapter topics, not alternate names for this code. See [module](../mlfirst/classical.py), [book PCA helper](../book_code/companion_classical.py), and [reconstruction tests](../tests/test_classical.py).
