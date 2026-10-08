# 5 · Optimization and generalization

**Prerequisites:** lessons 2–4, gradients and a validation split.
**Goal:** distinguish solving the training objective from predicting new outcomes.

## Follow a gradient carefully

Training selects parameters by minimizing a loss. For empirical risk
`J(theta) = mean(loss_i(theta)) + lambda*penalty(theta)`, the learning rate
controls how far an update travels. Changing a sum to a mean changes the balance
with the penalty unless its coefficient is adjusted.

Consider `J(w)=a*w^2/2`, with positive curvature a. Its derivative is a*w, so
gradient descent becomes `w_next=(1-rate*a)*w`. Convergence to zero requires
`abs(1-rate*a)<1`, or `0<rate<2/a`. With a=2, initial w=3 and rate=.25, the
sequence is 3, 1.5, .75, .375. At rate=1.1 the multiplier is -1.2: signs alternate
while magnitudes grow. A correct derivative can still produce bad updates.

Many-dimensional objectives have different curvature in different directions.
Standardizing features can make a single learning rate more effective. Momentum
and adaptive optimizers modify updates, but none makes every objective or data
choice sound. This experiment intentionally isolates the basic step mechanics;
it is not a full Adam or minibatch training framework.

## Fitting capacity is not evidence of generalization

We generate 18 noisy observations from a line and fit polynomial degrees 1, 5
and 15. A polynomial is linear in its coefficients: the design row is
`[1, x, x^2, ..., x^degree]`. Least squares solves for the coefficients directly,
so optimization is not the main difference between the candidates.

A high degree can follow small accidents in the training sample, especially
between observations and near boundaries. Training error may fall while
validation error increases. Validation uses independently generated noise at
101 inputs in the same interval. We select degree using validation RMSE, then
stop. That score is a tuning result; an honest final performance claim would
require an untouched test set or an outer validation procedure.

```bash
python -m mlfirst --lesson 5
python -m mlfirst --lesson 5 --seed 17
```

Inspect `stable_path` and `unstable_path`, then `degree_scores` and
`selected_degree`. Do not assume degree 1 wins every finite random sample.
The useful observation is how flexibility changes sensitivity, and whether the
pattern survives several seeds. This small polynomial example is not a universal
law that more parameters always make generalization worse.

## Practice

1. For curvature a=5, find the stable learning-rate interval. What happens at
   its upper boundary?
2. Double the training sample size while keeping noise and candidate degrees
   fixed. Predict which candidate's variance is most likely to shrink.
3. Why would repeatedly inspecting test scores to choose degree invalidate the
   claim that test data are untouched?

<details><summary>Hints and worked solutions</summary>

1. `0<rate<.4`. At .4 the multiplier is -1: the iterate oscillates without
   shrinking. At zero there is no progress.
2. More independent points generally constrain the flexible fit more strongly.
   Degree 15 is likely to become less sample-sensitive; this is an empirical
   tendency, not a guarantee for every sample or extrapolation problem.
3. The test outcomes influence model choice. Their final score then benefits
   from adaptation to that particular sample and acts as a validation score.

</details>

**Code:** [quadratic_descent and lesson_05](../mlfirst/foundations.py).
