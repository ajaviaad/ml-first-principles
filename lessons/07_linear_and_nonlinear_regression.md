# 07 — Linear and nonlinear regression

**Prerequisites:** vectors and matrices (lesson 03), mean squared error (06), and the idea of a training/test split (02). **Objective:** fit a line, explain its residuals, and distinguish curvature in the inputs from nonlinearity in the fitted coefficients.

A regression model predicts a numerical response. With `X` shaped `(n, p)`, coefficients `w` shaped `(p,)`, and a scalar intercept `b`, its prediction is `b + X @ w`. Ordinary least squares minimizes the **sum** of squared errors (SSE). Ridge minimizes `||y-b-Xw||² + λ||w||²`. This implementation leaves the intercept unpenalized. Changing the loss from a sum to an average changes the meaning of the same numerical λ.

For observations `x=[0,1,2]`, `y=[1,2,2]`, the mean input is 1 and mean response is 5/3. The slope is the centered cross-product divided by the centered input sum of squares: `1/2`. Thus the intercept is `5/3 - 1/2 = 7/6`. Residuals are `[-1/6, 1/3, -1/6]`, with SSE `1/6`. Their sum is zero. Their dot product with the input is also zero: the fitted residual is perpendicular to each design column.

Run from the repository root after the setup in the README:

```bash
python -m mlfirst --lesson 07
```

`coefficients`, `residuals`, `sse`, and `residual_sum` reproduce that calculation. `ridge_coefficients_lambda_2` shrinks the slope to 1/4 while moving the intercept to 17/12. The penalty does not force the intercept toward zero. The demo also constructs noisy quadratic training observations and evaluates a line and a quadratic on separate inputs within the training range. Compare `heldout_line_mse` with `heldout_quadratic_mse`. Those targets are noise-free known function values, so these metrics estimate function approximation error, not performance on future noisy observations.

A quadratic uses columns `[x, x²]` and still fits linearly in its coefficients. `polynomial_features` builds those columns; `ridge_fit` adds the intercept. You can use the functions independently:

```python
from mlfirst.classical import polynomial_features, ridge_fit, linear_predict
X = polynomial_features([-1, 0, 1], degree=2)
coef = ridge_fit(X, [2, 1, 2])
print(linear_predict(coef, polynomial_features([0.5], degree=2)))
```

Expect 1.25. The solver uses augmented least squares, avoiding an explicit inverse. Duplicate columns need not prevent predictions, but their individual coefficients become ambiguous. Ridge depends on feature units; scale using training information only. High-degree polynomials can overflow or extrapolate wildly. A lower in-range error does not validate behavior beyond the observed range or establish a causal effect.

**Practice**

1. Recalculate the ridge slope when λ is 6. Hint: the centered input sum of squares is 2.
2. Add 10 to every response. Which fitted quantities change?
3. Predict the quadratic above at `x=10`. Explain why its exact arithmetic is not evidence for a physical forecast.

<details><summary>Worked solutions</summary>

1. The slope is `1/(2+6)=1/8`; the intercept is `5/3-1/8=37/24`.
2. The intercept increases by 10. Slopes and residuals are unchanged because centering removes the shift and the intercept is free.
3. The fitted function is `1+x²`, giving 101. It is algebraically correct for the constructed function, but measurements restricted to `[-1,1]` cannot establish that real-world behavior continues quadratically.

</details>

**Scope and source:** this runnable lesson covers linear/ridge and polynomial regression. It does not implement lasso, elastic net, splines, Huber fitting, or Poisson regression discussed in the chapter. Read [the reusable module](../mlfirst/classical.py), [the book's exact companion](../book_code/companion_classical.py), and [independent checks](../tests/test_classical.py).
