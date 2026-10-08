# 32. An end to end investigation

This capstone forecasts demand in an entirely synthetic delivery business. Its purpose is to make an experiment's argument visible: what was fitted, what was selected, what was calibrated, and what evidence remained untouched until evaluation. No book, downloaded dataset, or external service is needed.

**Prerequisites:** least squares, validation, residuals, and basic uncertainty intervals. **Objectives:** maintain four distinct data roles, compare fixed models using paired losses, calibrate a finite-sample interval, and investigate changes in inputs separately from changes in the response mechanism.

## Describe the world before fitting

Temperature T is standardized using fixed constants `t=(T-18)/5`. Promotion p and weekend w are binary. The simulator uses

```text
mean = 80 + 7t - 2t^2 + 10p + 14w + 12pw
noise standard deviation = 3 + 2w
y = mean + Gaussian noise
```

At t=0, p=1, w=1, expected demand is `80+10+14+12=116`, and noise standard deviation is 5. The weekend group is intrinsically more variable. These are independent hypothetical days, not a calendar time series.

Three least-squares candidates use an intercept only, additive `[1,t,p,w]`, or expanded `[1,t,p,w,t^2,pw]` features. The expanded model is linear in its coefficients despite nonlinear input features. Training has 1,000 rows; validation, calibration, and test each have 250. Training fits coefficients; validation chooses the smallest RMSE; calibration sees the selected fixed model; final test assesses the completed procedure.

## Quantify prediction and comparison

Absolute calibration residuals give a split-conformal radius. For miscoverage alpha .1,

```text
rank = ceil((n_calibration + 1)*(1-alpha))
radius = calibration absolute residual at that one-based sorted rank
interval(x) = prediction(x) +/- radius
```

For 250 scores the rank is 226. If the rank exceeds the sample size, this conservative construction returns infinity. Do not silently cap the rank and claim the same guarantee. Exchangeability supports marginal coverage, not exact realized coverage or equal coverage in every subgroup.

To compare fixed predictors, calculate each test case's squared-loss difference: baseline loss minus candidate loss. Bootstrap these paired differences. Positive values favor the candidate. Pairing keeps the common case difficulty aligned. The resulting interval describes variation across cases conditional on fitted models; it excludes refitting uncertainty.

## Run and read two distinct experiments

```bash
python -m mlfirst --lesson 32 --seed 42
python book_code/companion_investigation.py
```

The first command includes `original_book_reproduction`, which directly executes the original program using data seed 20261004 and bootstrap seed 314159. Expect selected model `expanded`, test RMSE approximately 3.590, coverage .904, conformal radius approximately 5.749, and shifted coverage .812. The second command prints only that original reproduction.

`seeded_extension` uses your CLI seed, so its results differ. It adds subgroup counts and coverage, plus four controlled scenarios: fresh nominal data, input shift only, concept shift only, and both. These scenarios use common random numbers to isolate mechanisms; their metrics are dependent and are not four independent trials. Coefficients and calibration radius stay fixed.

```python
from mlfirst.decisions import investigation
result = investigation(seed=17)
print(result['test']['subgroups'])
print(result['controlled_scenarios']['concept_shift_only'])
```

Subgroup estimates can be noisy, especially for promoted weekends. Changed conditional outcomes invalidate the original exchangeability argument. A wider interval might improve coverage while becoming useless for capacity decisions. Adaptation requires representative labels, fresh calibration, and evaluation of practical costs.

## Exercises

1. Why must calibration labels not select among candidate models? Hint: calibration scores would become favorable evidence used twice.
2. Predict which subgroup suffers when the interaction rises from 12 to 24. Hint: inspect `p*w`.
3. What changes if the calibration sample has only two cases at alpha .1? Hint: calculate the rank.

<details><summary>Worked solutions</summary>

1. Selection based on calibration residuals compromises the separation supporting the usual split-conformal argument. Select using validation, then calibrate the fixed predictor independently.
2. Promoted weekends receive 12 additional demand units. Other cases have zero interaction change. A frozen model generally underpredicts that subgroup.
3. `ceil(3*.9)=3`, beyond two available scores, so the radius is infinite. More calibration data or a less demanding coverage target is needed for a finite interval here.

</details>

Sources: [the original executable investigation](../book_code/companion_investigation.py) and [a primary conformal-prediction tutorial](https://arxiv.org/abs/2107.07511).
