# 2 · Data and experiments

**Prerequisites:** lesson 1, array rows and columns.
**Goal:** keep shared information out of held-out evaluation and fit preprocessing
only where training is permitted.

## A split represents future use

Four recordings from the same microphone share hardware and background noise.
If future predictions must handle unseen microphones, splitting random recordings
can make the test unrealistically easy. A grouped split assigns each microphone
entirely to one side. Our example has 12 groups with four rows each. The test
fraction applies to the number of groups; groups of unequal sizes would give a
different fraction of rows.

This differs from stratification, which seeks similar class proportions. Neither
one substitutes for the other. For future forecasting, chronology may be the
right boundary. A gap is appropriate when labels or input availability windows
overlap the split. It should follow the application, not an arbitrary convention.

Keep training, validation and test roles distinct. Fit parameters on training;
choose features, hyperparameters and thresholds with validation; assess the
chosen procedure on an untouched test set. Repeatedly changing a model after
seeing test performance converts that set into development data.

## Fit a transformation

Standardization computes `z_ij = (x_ij - mean_j) / scale_j`, where the mean and
population standard deviation are estimated on training rows. If training values
are `[1, 3]`, mean is 2 and scale is 1, so they become `[-1, 1]`. A later value 5
becomes 3; it need not fall in the training range. Recomputing statistics on the
later sample would change the meaning of the coordinate system.

The `Standardizer` stores vectors of shape `(d,)` and transforms matrices of
shape `(n, d)`. Constant columns receive scale 1 and become zero on training
data. Nonfinite values are rejected rather than silently imputed. If imputation
is needed, estimate its parameters inside the same training boundary.

## Run and interpret

```bash
python -m mlfirst --lesson 2
```

`shared_groups` should be empty. The explicit `train_indices` and `test_indices`
let you reconstruct the assignment. `train_scaled_mean` is approximately zero;
`test_scaled_mean` can differ because it is not independently centered.
`transform_changed_fit` must be false. The second feature is constant, exposing
the zero-variance case. `chronological_example` illustrates a separate ordered
split with a gap; it is not combined with the group experiment.

Seeds help repeat shuffling, but row order, software and data versions also
matter. Save actual split assignments in real investigations. A pipeline should
fit preprocessing inside each training fold, including feature selection and PCA.

## Practice

1. Training values are `[2, 6]`; the test value is 10. Standardize all three using
   only training statistics.
2. Use groups with sizes 1, 1, 1 and 20. Why can a 25% group holdout contain far
   more than 25% of the rows?
3. A target is sales during the next seven days. Explain why adjacent examples
   at a time split can leak even if every input timestamp is earlier.

<details><summary>Hints and worked solutions</summary>

1. Mean 4 and population standard deviation 2 give `[-1, 1]` and test value 3.
2. Selecting the group with 20 rows yields `20/23` of all rows. The unit sampled
   is a group. Evaluation design must consider both group and row counts.
3. The final training labels can include outcomes inside the validation period.
   Purge overlapping label windows or leave an appropriate gap. Merely sorting
   the input rows is insufficient when targets themselves look into the future.

</details>

**Code:** [Standardizer and split_groups](../mlfirst/foundations.py).
Continue with [the mathematical language](03_mathematical_language.md).
