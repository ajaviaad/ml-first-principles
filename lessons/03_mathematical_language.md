# 3 · The mathematical language of models

**Prerequisites:** arithmetic, the Python primer, lesson 2.
**Goal:** translate formulas into shape-safe array operations and check a gradient.

## Objects before symbols

A scalar is one number, a vector is an ordered list and a matrix has rows and
columns. In this repository an input matrix `X` usually has shape `(n, d)`: n
examples and d features. A weight vector `w` has shape `(d,)`; `X @ w + b` produces
a prediction vector `(n,)`, adding scalar offset b to each entry. The `@` operator
performs matrix multiplication, while `*` is elementwise multiplication.

For row `[2, 3]`, weights `[4, -1]` and offset 7, the prediction is
`2*4 + 3*(-1) + 7 = 12`. The full book array example yields `[12, 6, 21]`.
Try computing these before running the code. This operation appears again in
linear models, neurons and attention.

NumPy broadcasting expands compatible dimensions. It is convenient but dangerous:
subtracting shape `(n, 1)` from `(n,)` produces `(n, n)`, not `(n,)`. Such code
may run and optimize the wrong objective. Check intended shapes explicitly.

## A derivative predicts a local change

For one observation x=2, y=5, define
`J(w,b) = 0.5*(w*x+b-y)^2`. At w=1, b=0, prediction is 2 and residual is -3.
The gradient is `[(prediction-y)*x, prediction-y] = [-6, -3]`. A step of 0.1
opposite this gradient gives `[1.6, 0.3]`, prediction 3.5 and loss 1.125 instead
of 4.5. The chain rule multiplies the derivative along each dependence path.

A central finite difference estimates coordinate j as
`[J(theta+eps*e_j) - J(theta-eps*e_j)] / (2*eps)`. It independently checks the
analytic derivative on small smooth problems. Tiny epsilon magnifies rounding;
large epsilon stops approximating a local change. It is a check, not an efficient
training method.

## Stable arithmetic

Exponentiating scores 1000 and 999 overflows ordinary double precision. Instead,
`logsumexp(z) = max(z) + log(sum(exp(z-max(z))))`. The shifted exponentials are
1 and about 0.368, while the mathematical result stays the same. Avoiding overflow
often means rearranging algebra, not lowering accuracy or clipping arbitrarily.

```bash
python -m mlfirst --lesson 3
python book_code/chapter03_array_example.py
```

Inspect `prediction`, `analytic_gradient`, `numeric_gradient`, `gradient_max_error`
and `loss_before`/`loss_after`. The gradient error should be near floating-point
precision at this scale, and `stable_logsumexp_1000_999` about 1000.3133. A check
at one point supports that calculation; it does not establish every case.

## Practice

1. Predict the shapes of `(4,3) @ (3,2)` and `(4,1) - (4,)`.
2. Differentiate `J(w)=3*w^2` at w=2 and verify with `central_difference`.
3. What is `logsumexp([1000,1000])` without computing `exp(1000)`?

<details><summary>Hints and worked solutions</summary>

1. The product has shape `(4,2)`; the subtraction broadcasts to `(4,4)`.
2. The derivative is 6w=12. Use
   `central_difference(lambda a: 3*a[0]**2, [2.0])`.
3. `1000 + log(2)`, about 1000.6931. Both shifted exponentials equal 1.

</details>

**Code:** [gradient and stability functions](../mlfirst/foundations.py) and
[the unchanged book example](../book_code/chapter03_array_example.py).
