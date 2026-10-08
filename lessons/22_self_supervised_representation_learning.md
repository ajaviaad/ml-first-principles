# 22. Self supervised representation learning

**Prerequisites:** vectors, cosine similarity, softmax, and cross-entropy. **Objective:** calculate a contrastive objective, inspect collapsed representations, and understand why the definition of a positive pair is part of the learning problem.

Self-supervised learning creates prediction targets from the observations themselves. A contrastive approach constructs two views of each item and encourages their representations to match. Different rows in a batch act as alternatives. A useful view transformation should preserve the information required by the intended downstream task. An augmentation that removes a tiny diagnostic feature may teach the representation to ignore exactly what matters.

Our experiment constructs twelve normalized vectors with eight coordinates and makes a slightly perturbed normalized second view of each. The similarity matrix is `S = Z1 @ Z2.T`, with shape `[12,12]`. Entry `S[i,j]` is a cosine similarity. Each diagonal entry is the intended match. The one-direction InfoNCE objective is

`L = mean_i(-log(exp(S[i,i]/tau) / sum_j(exp(S[i,j]/tau))))`.

Temperature `tau>0` controls how sharply score differences affect probabilities. The derivative with respect to the unscaled score matrix is `(softmax(S/tau)-I)/(n*tau)`. The helper returns this derivative; differentiating through an encoder would additionally require the derivatives of matrix multiplication and normalization. This experiment constructs features and evaluates the objective; it does not train an encoder.

For a hand calculation, take one positive similarity `.8` and two alternatives `.4` and `0`, with temperature `.2`. The logits are `[4,2,0]`, and the loss is `log(1+exp(-2)+exp(-4))≈.142932`. Subtracting the largest logit before exponentiating keeps the calculation stable. Multiplying every vector's norm would affect dot products; normalizing first removes that scale degree of freedom.

Run:

```bash
python -m mlfirst --lesson 22
```

Inspect `paired_info_nce`, `shuffled_pair_info_nce`, `collapsed_info_nce`, `pair_retrieval_accuracy`, and `hand_example_loss`. With seed 42, the paired loss is about `.288`, while incorrect pairing raises it to about `5.20`. If every feature is identical, all scores tie and the loss is `log(12)≈2.485`. This collapse reference helps interpret the objective, although a good batch loss alone does not establish useful downstream features.

False negatives are an important limitation: two different examples may depict the same underlying concept. Conversely, pairs can share camera or background artifacts that let an encoder solve the training task without learning the intended content. Evaluate a representation on a heldout task, use splits that prevent duplicate leakage, and compare with simple baselines. Retrieval on these explicitly constructed noisy copies is only a mechanism check.

## Exercises

1. Derive the collapsed loss for a batch of five identical embeddings.
2. What happens to the probability of the largest score as temperature approaches zero?
3. Can adding a constant to all scores in one row change its loss?

<details><summary>Hints and worked solutions</summary>

1. Every candidate has probability `1/5`, so the loss is `-log(1/5)=log(5)`.
2. With a unique maximum, its probability approaches one. If that maximum is an incorrect match, the loss becomes very large rather than improving.
3. No. The exponential factor cancels from numerator and denominator. The positive logit and log-normalizer shift equally. The score-gradient row therefore sums to zero.

</details>

Source: [`info_nce`, `normalize_rows`, and `lesson_22`](../mlfirst/neural.py). Verification: [`test_neural.py`](../tests/test_neural.py).
