# 08 — Probability for classification

**Prerequisites:** conditional probability (04), gradients (05), and classification metrics (06). **Objective:** distinguish a probability model from a threshold decision and compare direct conditional modeling with a model of feature evidence.

Logistic regression forms `z=b+Xw`, then `p=1/(1+exp(-z))`. `X` has shape `(n,p_features)` and the resulting probability vector has shape `(n,)`. A weight changes the log odds, so a unit increase multiplies odds by `exp(weight)`; it does not add that weight to probability. With `z=-2+0.8x`, probabilities at 0 and 1 are approximately 0.119 and 0.231. The odds ratio is `exp(0.8)≈2.226`.

Our objective is **mean** binary cross-entropy plus `λ||w||²/2`, with an unpenalized intercept. For a design matrix including a column of ones, the unregularized gradient is `design.T @ (p-y)/n`. The code uses `logaddexp(0,z)-y*z` rather than logarithms of rounded probabilities. Stable arithmetic matters for large scores, where naive exponentiation can overflow.

```bash
python -m mlfirst --lesson 08
```

The lesson generates labeled observations from a known logistic probability, fits the first 120 rows, and evaluates the last 40. `initial_loss` and `final_loss` are the penalized training objectives. `coefficients` contains the intercept first. `test_accuracy` measures thresholded decisions at 0.5; `test_brier_score` is the mean squared error of probabilities against binary outcomes. They answer different questions. The tiny test set is a demonstration, not a reliable model ranking or calibration study. `sigmoid_extremes` demonstrates stable values for scores -1000, 0, and 1000.

Naive Bayes instead combines class priors with class-conditional feature probabilities. The archive example has urgent prior 1/4, deadline probability 0.8 versus 0.1, and attachment probability 0.6 versus 0.2. For both features present, urgent mass is `0.25×0.8×0.6=0.12`; nonurgent mass is `0.75×0.1×0.2=0.015`. Normalizing gives `archive_urgent_probability=8/9`.

The Bernoulli implementation also uses absent features: their likelihood is `1-p_feature`. `bernoulli_nb_fit` smooths each feature with `(count+alpha)/(class_count+2*alpha)` and uses empirical class priors. Model feature probabilities have shape `(2,p_features)`; predictions have shape `(n,2)`. Classes are explicitly 0 and 1.

The independence assumption is conditional on the class. Duplicating the deadline feature adds no information but multiplies the likelihood ratio by eight again. `archive_duplicate_deadline_probability=64/65` demonstrates unjustified confidence from double counting. Logistic regression has its own failure modes: complete separation drives unpenalized coefficients outward, and an excessive learning rate can prevent descent. Always inspect the recorded objective and evaluate fresh data.

**Practice**

1. Compute the urgent probability when neither feature is present. Hint: use complements.
2. A class contains two examples with feature values `[0,1]`. What is its smoothed feature probability with alpha 1?
3. Change a decision threshold from 0.5 to 0.8. Does that refit the probability model?

<details><summary>Worked solutions</summary>

1. Urgent mass is `.25×.2×.4=.02`; nonurgent mass is `.75×.9×.8=.54`. The posterior is `.02/.56=1/28`.
2. `(1+1)/(2+2)=1/2`. Smoothing prevents exact zero and one estimates in small classes.
3. No. Fewer cases are labeled positive, generally reducing recall and changing precision. The underlying probabilities and Brier score stay the same.

</details>

**Scope and source:** binary logistic and Bernoulli naive Bayes are implemented. Multinomial/Gaussian naive Bayes, multiclass fitting, and prior-shift correction are chapter concepts, not supplied estimators. See [implementation](../mlfirst/classical.py), [original companion](../book_code/companion_classical.py), and [gradient and arithmetic tests](../tests/test_classical.py).
