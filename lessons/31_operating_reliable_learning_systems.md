# 31. Operating reliable learning systems

A model is one component of a system that receives inputs, produces outputs, and influences decisions. Correct matrix multiplication cannot rescue a feature supplied in the wrong units. This lesson builds small, inspectable monitoring and explanation tools, then states what those tools can and cannot establish.

**Prerequisites:** means, standard deviations, regression error, and held-out evaluation. **Objectives:** distinguish input drift from predictive damage, measure missingness, compute permutation importance, and explain a linear prediction relative to a declared baseline.

## Monitor an explicit contract

A prediction contract specifies input names, types, units, ranges, missing-value handling, timing, and output meaning. Preserve fitted preprocessing with model parameters. Record the training population, feature version, evaluation evidence, and intended use. A monitor needs an action: investigate a data source, use a documented fallback, or begin a retraining experiment with representative labels.

The example compares each current feature mean with a reference distribution:

```text
standardized_mean_shift = (current_mean - reference_mean) / reference_std
missing_fraction = missing_count / total_count
```

If reference values are `[0,2]`, their mean is 1 and population standard deviation is 1. Current values `[3,3]` therefore have standardized mean shift 2. A constant reference column has zero standard deviation; the code flags it and uses scale one, rather than dividing by zero. An entirely missing current column has no defined mean shift and returns `None`.

Input changes can be detected before labels arrive. They do not establish error deterioration. Covariate shift changes inputs while preserving the conditional target relationship; concept drift changes that relationship. Label shift changes class frequencies while preserving class-conditional input distributions. Real changes may combine these mechanisms. Delayed and selectively observed labels require additional care before interpreting residuals.

## Inspect model behavior

Permutation importance evaluates a fixed predictor on held-out cases, shuffles one input column, and measures the change in error:

```text
importance_j = MSE_after_permuting_j - original_MSE
```

The implementation repeats permutations and reports their standard deviation. That variation describes random disruption on this dataset; it is not a confidence interval for importance in every future population. Correlated features may substitute for each other, and independent shuffling can create unrealistic rows.

For a linear predictor, an exact additive explanation relative to baseline b is

```text
prediction = (intercept + beta @ b) + sum_j beta_j*(x_j-b_j)
```

With intercept 1, beta `[4,-2]`, input `[2,3]`, and baseline `[1,1]`, baseline prediction is 3, contributions are `[4,-4]`, and prediction remains 3. Different baselines change the allocation. This decomposes a model output; it does not identify causal effects.

## Run and interpret

```bash
python -m mlfirst --lesson 31 --seed 42
```

Feature zero has an introduced mean shift. Feature one has 10% missing current values. Held-out permutation importance should identify feature zero as most influential, feature two as weaker, and feature one as zero for the constructed predictor. `partial_dependence_feature_0` averages predictions with feature zero set to -1, 0, and 1. Inspect plausibility before interpreting such replacements on real data.

```python
from mlfirst.decisions import drift_report, linear_explanation
print(drift_report([[0,1],[2,1]], [[3,1],[3,1]]))
print(linear_explanation([2,3], [4,-2], intercept=1, baseline=[1,1]))
```

All executable results here are new synthetic exercises accompanying the chapter. They implement no production alert threshold, privacy guarantee, latency benchmark, or automated retraining policy.

## Exercises

1. Why can input drift occur without predictive damage? Hint: a correct linear relationship may remain valid over the new range.
2. What does negative permutation importance mean? Hint: disrupting an input can occasionally improve held-out error.
3. What remains unchanged when only the explanation baseline changes? Hint: distinguish allocation from prediction.

<details><summary>Worked solutions</summary>

1. Inputs can become more common in another region while the learned relationship remains accurate there. Representative labeled evaluation is needed to judge damage.
2. The disrupted dataset scored better in this experiment. Sampling variation, overfitting, or feature dependence can cause this; it does not imply a universally harmful real-world variable.
3. The prediction remains unchanged. Baseline prediction and contributions move in offsetting amounts, provided the decomposition is recomputed consistently.

</details>

Read [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems) for a broader account of dependencies and feedback loops around deployed models.
