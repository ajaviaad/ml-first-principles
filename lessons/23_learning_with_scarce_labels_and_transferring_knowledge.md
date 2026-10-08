# 23. Learning with scarce labels and transferring knowledge

**Prerequisites:** linear regression or classification, regularization, and matrix multiplication. **Objective:** compare a small learned head on fixed features with a raw-feature baseline, then verify a low-rank parameter update. This lesson uses designed features; it does not download or claim to reproduce a pretrained foundation model.

When labels are scarce, a useful existing representation can reduce the task-specific learning burden. Freeze that representation and fit a small head. This makes the assumptions visible: features must preserve the property needed by the new labels. Good source-task performance alone does not prove that they do.

Our synthetic task labels a point positive when its coordinates have the same sign. A linear boundary in the original two coordinates cannot express the entire rule. The fixed feature `x1*x2` exposes it directly. We learn a ridge regression head using only 24 labeled points and evaluate its sign on 400 separately generated points. A raw linear baseline uses the same labels and test set. The fixed representation deliberately contains the correct interaction, so the comparison illustrates favorable representation choice rather than measuring real transfer learning.

If `F[n,k]` contains features and `y[n]` contains targets `-1` or `+1`, the fitted head is

`w = solve(F.T @ F + lambda*I, F.T @ y)`.

For this small example, all coefficients, including the intercept, are regularized with `lambda=.01`. Predictions threshold `F @ w` at zero. Finite data can shift the boundary even with a useful feature, so test accuracy need not be perfect.

The second mechanism is low-rank adaptation. For frozen weight `W[d,o]`, matrices `A[d,r]` and `B[r,o]` define

`Y = X @ W + scale * (X @ A) @ B`.

The update has rank at most `r`. Materializing `W + scale*A@B` gives the same answer. With `d=8,o=5,r=2`, the factors have `2*(8+5)=26` trainable parameters versus 40 for a dense update. This function verifies the algebra; it does not optimize the factors.

Run:

```bash
python -m mlfirst --lesson 23
```

Inspect `frozen_feature_accuracy`, `raw_linear_accuracy`, `labeled_training_examples`, `lora_materialization_error`, and the two parameter counts. Seed 42 gives approximately `.89` versus `.48` heldout accuracy. The materialization error is near floating-point precision. Small rank reduces parameter count only when `r*(d+o) < d*o`; it can also restrict the adaptations the model can express.

A major pitfall is using test labels to choose features or tuning settings, then calling the resulting score heldout evidence. Pseudo-labeling creates another risk: high confidence can reinforce a wrong boundary. Negative transfer occurs when inherited features or constraints obstruct the new task. Always retain a simple target-only baseline.

## Exercises

1. Calculate the low-rank parameter count with `d=8,o=5,r=4`.
2. What does setting `scale=0` do?
3. Would the interaction feature alone solve a target depending on `x1+x2`?

<details><summary>Hints and worked solutions</summary>

1. The count is `4*(8+5)=52`, greater than a 40-parameter dense update. Low rank is not automatically cheaper at every size.
2. The result becomes exactly `X @ W`; the learned correction contributes nothing.
3. No. Points with identical products can have different sums, so this compression discards information needed for that target. Add appropriate features or adapt the representation.

</details>

Source: [`lora_apply` and `lesson_23`](../mlfirst/neural.py). Verification: [`test_neural.py`](../tests/test_neural.py).
