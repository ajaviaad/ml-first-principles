# 6 · Measuring what matters

**Prerequisites:** probability, independent validation, vector predictions.
**Goal:** connect a score to a decision and distinguish ranking, calibration and
coverage.

## Count the errors that matter

In 1,000 requests, 50 are urgent. A system flags 80 requests, including 40 urgent
ones. Then TP=40, FP=40, FN=10 and TN=910. Precision is `TP/(TP+FP)=.5`, recall
is `TP/(TP+FN)=.8`, and accuracy is `(TP+TN)/n=.95`. The high accuracy coexists
with missing one in five urgent requests. Always identify the positive class.

F1 is `2*TP/(2*TP+FP+FN)`. It summarizes precision and recall, but does not
automatically express actual error costs. Our implementation reports undefined
ratios as `None` (JSON null), rather than silently claiming zero or perfection.

If false positives cost 1 and false negatives cost 9, a probability-based decision
compares `1*(1-p)` against `9*p`. Predict positive at a threshold of .1 when
correct decisions cost zero. This threshold is justified for relevant conditional
probabilities under those costs. The synthetic scores here are deliberately
imperfect; applying the formula does not establish that they are calibrated.

## Probabilities and rankings answer different questions

ROC AUC measures pairwise ordering: the chance a positive outranks a negative,
with half credit for ties. It does not certify probability quality. Brier score
averages `(p-y)^2`; log loss penalizes confident wrong forecasts more severely.
The readable pairwise AUC implementation has quadratic cost and is suitable for
small examples, not large production datasets.

For regression, MAE uses absolute errors; RMSE emphasizes larger errors. Both
have target units. R² compares squared error against a mean reference and can
be negative on held-out data. A constant target leaves its usual denominator
zero, represented as `None` here.

## Calibrate a prediction interval

With m independent calibration residuals and desired coverage `1-alpha`, take
the sorted absolute residual at rank `ceil((m+1)*(1-alpha))`. A prediction becomes
`[prediction-radius, prediction+radius]`. For m=19 and alpha=.1, rank is 18.
If the residuals are `.2, .4, ..., 3.8`, the radius is 3.6 and width is 7.2.
For too little calibration data, the required rank can exceed m; the valid
construction returns an infinite radius rather than inventing a finite guarantee.

```bash
python -m mlfirst --lesson 6
```

Compare `threshold_0_5` and `cost_sensitive` counts, then inspect
`conformal_rank_for_19_at_90_percent`, `conformal_radius` and `conformal_width`.
Ordinary split conformal coverage is marginal under exchangeability, not a
promise for every individual or subgroup. Report subgroup size, coverage and
width separately when those decisions matter. Lesson 32 applies this workflow
to a complete investigation and distribution shift.

## Practice

1. Calculate F1 from TP=40, FP=40 and FN=10.
2. Find the decision threshold if false positives cost 4 and false negatives 1.
3. With m=2 and alpha=.1, calculate the conformal rank and explain the result.

<details><summary>Hints and worked solutions</summary>

1. `80/(80+40+10)=8/13`, about .6154.
2. `4/(4+1)=.8`, assuming the stated costs and appropriate probabilities.
3. `ceil(3*.9)=3`, beyond the two residuals. The radius is infinite; the data
   cannot support a finite interval through this construction at that level.

</details>

**Code:** [metrics and conformal_radius](../mlfirst/foundations.py).
