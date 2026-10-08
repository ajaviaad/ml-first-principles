# 1 · Learning problems and evidence

**Prerequisites:** basic arithmetic; the [Python primer](../docs/GETTING_STARTED.md).
**Goal:** define what a prediction means, establish a baseline, and distinguish
the quantity you want from the quantity your records actually contain.

## A model begins with a question

Suppose a bakery chooses its production quantity before opening. An example is
one day. Its input is whether the day is a weekend, known before production;
its target is the day's unconstrained demand. The output is a numerical forecast,
so this is supervised regression. Replacing the target with “will demand exceed
100?” creates classification. Looking for groups of similar days without targets
would be unsupervised learning. Learning prices that change future customer
behavior would require a sequential decision problem.

The target, available information and decision time are part of the problem.
Adding today's final sales to a morning forecast would reveal future information.
Making that feature available in a spreadsheet does not make it available in use.

A constant baseline predicts the mean of the **training** targets. Our slightly
richer model estimates separate weekday and weekend means. Both are fitted using
the first 280 days and evaluated on the following 140. The test period does not
choose the groups or fit their means. This tiny setup assumes the mechanism stays
stable over time; later lessons deliberately break that assumption.

## Work one example

For training demands `[70, 90, 110, 130]`, the constant mean is 100. If the first
two are weekdays and the last two weekends, the two group means are 80 and 120.
For future demands `[82, 118]` on a weekday and weekend, respectively, the
constant's absolute errors are 18 and 18; the group model's are 2 and 2.

Mean absolute error is `MAE = sum(abs(prediction - target)) / n`. The vector of
targets and predictions must each have shape `(n,)`. A smaller MAE on appropriate
held-out cases supports a narrower prediction claim, not a causal explanation
or universal superiority.

## Run and interpret

```bash
python -m mlfirst --lesson 1
```

Compare `constant_baseline` and `weekday_model`. Their MAE and RMSE measure the
same test cases in loaves. `training_means_by_weekend` contains the two fitted
parameters. In this constructed world weekend demand is 30 loaves higher, so
the group feature should be useful. Noise prevents perfect prediction.

Now compare `mean_demand` with `mean_observed_sales_if_stock_90`. Sales are
`min(demand, stock)`: customers who arrive after stock runs out are absent from
the sales record. This is censoring. A model can accurately predict the sales
record while underestimating the demand the bakery wanted to forecast.

## Practice

1. Calculate MAE for targets `[10, 20, 30]` and predictions `[20, 20, 20]`.
2. Change the synthetic weekend effect from 30 to zero. Predict what happens to
   the advantage of the group model, then compare several seeds.
3. Would a perfect predictor of historical sales identify the effect of raising
   prices? State the additional evidence needed.

<details><summary>Hints and worked solutions</summary>

1. Absolute errors are 10, 0 and 10, giving `20/3`, about 6.67.
2. Without a weekend effect, separate means estimate the same population mean
   from fewer samples. Their extra sampling variation can make prediction worse.
   A single random seed may still give a small accidental improvement.
3. No. Historical associations may reflect season, availability or customer
   selection. An intervention study or defensible causal assumptions are needed
   to infer a price effect. Prediction accuracy alone does not establish it.

</details>

**Read the implementation:** [lesson_01 and regression_metrics](../mlfirst/foundations.py).
Continue with [data and experiments](02_data_and_experiments.md).
