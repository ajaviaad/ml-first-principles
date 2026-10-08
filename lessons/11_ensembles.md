# 11 — Ensembles

**Prerequisites:** squared-error regression (07), trees (10), and sampling (04). **Objective:** distinguish averaging independently resampled fits from fitting successive residual corrections, and see why diversity matters.

An ensemble combines several models. Bagging draws bootstrap samples of the training rows, fits a separate model to each draw, and averages predictions. Each draw has the original sample size but permits repetition. With many observations, about 63.2% of distinct training rows appear in a typical draw; the rest can be used for out-of-bag evaluation. This implementation returns the draw indices so the mechanism is inspectable.

For component errors with common variance `σ²` and common pairwise correlation `ρ`, the average of `B` models has variance `σ²(ρ+(1-ρ)/B)`. With `σ²=1`, `ρ=.2`, and `B=20`, this is .24. Adding models cannot eliminate the shared .2 component. Independence is an idealized assumption, and real ensemble members can have different variances and correlations.

```bash
python -m mlfirst --lesson 11
```

`ensemble_size` is 20. `first_bootstrap_indices` shows one draw; repetitions are permitted but do not occur in every draw. `mean_bootstrap_unique_fraction` summarizes distinct coverage in these four-row draws, so do not expect the large-sample 63.2% approximation exactly. `bagging_predictions` averages depth-one regression trees fitted on those samples. Results are deterministic for a fixed seed. `variance_formula_rho_0_2_B_20` reports .24 from the theoretical example; it is not an empirically measured variance of this fitted ensemble.

Boosting is sequential. For squared loss, the negative gradient with respect to predictions is proportional to the residual `y-prediction`. Fit a small tree to residuals, then add a shrunken correction. Our update is `prediction += learning_rate * stump_prediction`; the initial prediction is the training mean. All arrays of targets, residuals and predictions have shape `(n,)`.

The four targets `[2,2,6,6]` start at mean 4. Residuals are `[-2,-2,2,2]`, giving SSE 16. A stump splits the low and high targets perfectly. At learning rate .5, the correction is `[-1,-1,1,1]`, giving predictions `[3,3,5,5]` and SSE 4. A second correction produces `[2.5,2.5,5.5,5.5]` and SSE 1. Inspect `boosting_sse_history=[16,4,1]` and `boosting_predictions`.

The demo uses training SSE to expose arithmetic, not to select the number of stages. With noisy data, more stages can overfit even while training error falls. Tune stage count and tree complexity on validation data, keeping test data untouched. Bagging often reduces variance of unstable learners; it does not automatically remove bias. Repeated or dependent observations require group-aware resampling rather than blindly treating rows as independent.

**Practice**

1. Add a third boosting stage with learning rate .5. Predict the SSE before running it.
2. What variance does the formula approach as the number of ensemble members grows without bound?
3. For the bootstrap indices `[0,0,2,3]`, which training row is out of bag and what fraction of unique rows was included?

<details><summary>Worked solutions</summary>

1. Each stage halves every residual here, so squared error is quartered: `1/4`. This exact pattern depends on the stump fitting the residual structure perfectly.
2. It approaches `ρσ²`, here .2. Shared errors persist under averaging.
3. Row 1 is absent. Three of four unique rows appear, so the fraction is .75. That absent row can evaluate this particular tree without being in its training draw.

</details>

**Scope and source:** bootstrap regression trees and squared-loss stump boosting are implemented. Random feature selection, a full random forest, AdaBoost, multiclass boosting, and stacking are not. See [ensemble code](../mlfirst/classical.py), [book boosting arithmetic](../book_code/companion_classical.py), and [tests](../tests/test_classical.py).
