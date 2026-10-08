# 16. Neural networks and backpropagation

**Prerequisites:** matrix multiplication, the chain rule, gradient descent, and binary cross-entropy. **Objective:** follow derivatives through a two-layer network, verify them numerically, and fit the four points of XOR. This guide stands alone; it also accompanies Chapter 16 of the book.

A dense layer transforms a row-major batch `X[n,d]` using `Z = X @ W + b`, where `W[d,h]` and `b[h]`. The hidden representation is `H = tanh(Z)`. A second matrix `V[h,1]` and bias produce one logit per example. Applying a sigmoid gives a probability. Without the nonlinear activation, composing the two affine layers would still produce only one affine transformation.

For labels `y[n,1]`, the mean loss is

`L = mean(logaddexp(0, logits) - y * logits)`.

This form avoids directly taking the logarithm of a rounded probability. The derivative arriving at the logits is `(sigmoid(logits) - y) / n`. Consequently `dV = H.T @ error`; the derivative through tanh multiplies by `1 - H**2`. The first layer receives `dW = X.T @ hidden_error`. Compute every derivative before updating any parameter so all derivatives describe the same forward calculation.

A hand calculation makes that last rule concrete. Let `x=2`, `w=.5`, `b=0`, `v=3`, and `c=-1`, with a ReLU hidden unit and target `1`. The prediction is `2`, and half-squared loss is `.5`. Derivatives with respect to `(w,b,v,c)` are `(6,3,1,1)`. One simultaneous step of size `.01` produces prediction `1.5315`. This scalar example uses a different loss and activation from the XOR experiment; it illustrates the same chain rule.

Run from the repository root:

```bash
python -m mlfirst --lesson 16
```

Inspect `initial_loss`, `final_loss`, `xor_probabilities`, `gradient_error`, and `training_accuracy`. With seed 42 the loss falls from about `.729` to `.00225`, and all four points are classified correctly. `gradient_error` compares analytic derivatives against central differences, using double precision. Tiny error checks the implementation locally. Perfect accuracy on four training points establishes a fit, not evidence about new real-valued inputs.

The main pitfalls are missing the batch-average factor, silently transposing the wrong dimension, and using an updated downstream weight during an upstream calculation. Finite differences also depend on step size: too small invites cancellation; too large measures curvature over an interval. ReLU kinks require additional care, which is why the automated check uses smooth tanh.

## Exercises

1. Replace the nonlinear hidden activation with identity. Can this classifier learn XOR?
2. Change the batch loss from a mean to a sum. What happens to gradients?
3. Derive the first-layer bias gradient.

<details><summary>Hints and worked solutions</summary>

1. Compose the two affine maps. The result is affine, so its linear boundary cannot separate the XOR corners. More affine layers do not change this.
2. Summation removes division by `n`; every gradient becomes `n` times larger. Divide the learning rate by `n` to preserve the same update.
3. Each example adds the same bias vector before tanh. Sum the hidden preactivation derivatives over examples: `hidden_error.sum(axis=0)`.

</details>

Source: [`lesson_16`](../mlfirst/neural.py), [`mlp_loss_grad` and `gradient_check`](../book_code/companion_deep.py). Verification: [`test_neural.py`](../tests/test_neural.py).
