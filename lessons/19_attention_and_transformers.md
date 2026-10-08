# 19. Attention and transformers

**Prerequisites:** matrix multiplication, weighted averages, and probability normalization. **Objective:** calculate one attention lookup, enforce a causal mask, and distinguish this mechanism from a complete transformer. You can follow the experiment without knowing recurrent networks.

Attention allows one representation to retrieve a weighted mixture of other representations. Queries specify the requests, keys determine matching scores, and values contain the delivered content. A single scaled dot-product head is

`A = softmax(Q @ K.T / sqrt(d)); output = A @ V`.

The softmax operates across keys separately for each query. With `Q[nq,d]`, `K[nk,d]`, and `V[nk,dv]`, scores and weights have shape `[nq,nk]`; the output has shape `[nq,dv]`. Query count and key count need not match. In cross-attention they may come from different streams. In self-attention they are derived from the same input sequence.

Compute one lookup by hand. Let `Q=[1,0]`, `K=[[1,0],[0,1]]`, and `V=[[10],[2]]`. The scaled scores are `[1/sqrt(2),0]`. Their softmax is approximately `[.669762,.330238]`, so the returned value is `.669762×10 + .330238×2 ≈ 7.358092`. Changing a value changes the retrieved content without directly changing these weights. Changing a key changes where attention goes.

For causal generation, position `t` must not read future positions. The code receives a Boolean `allowed[nq,nk]` mask and replaces prohibited scores with negative infinity before softmax. A lower-triangular mask permits the current token and earlier tokens. Every query must retain at least one key; otherwise its normalizing sum is undefined. The helper checks and rejects that case.

Run:

```bash
python -m mlfirst --lesson 19
```

Inspect `weights`, `weighted_value`, `attention_score_shape`, and `causal_prefix_error`. The experiment changes the last value vector by 1000 and checks that preceding causal outputs remain unchanged. This is a targeted information-flow test: a correct shape alone would not establish the absence of future leakage.

A transformer usually adds learned projections, several heads, residual paths, normalization, position information, and positionwise feedforward layers. None of those is silently claimed by this one-head demonstration. Dense attention stores a query-by-key score matrix; doubling both sequence lengths multiplies its entries by four. Masking information does not automatically avoid computing or storing all those entries in a straightforward implementation.

## Exercises

1. Set all query-key scores to zero. What does the output become?
2. Permit only the first key in the numerical example.
3. Add the same constant to every score in one row. Does its softmax change?

<details><summary>Hints and worked solutions</summary>

1. Equal exponentials give equal weights, so the output becomes the arithmetic mean of the value vectors. For two values 10 and 2, it is 6.
2. The normalized weights become `[1,0]` and the output is exactly 10. Zeroing an unwanted value instead would not remove its probability mass.
3. Multiplying every exponential by the same positive factor cancels between numerator and denominator. This explains why subtracting the maximum score improves numerical stability without changing the mathematical distribution.

</details>

Source: [`lesson_19`](../mlfirst/neural.py), [`attention`](../book_code/companion_deep.py). Verification: [`test_neural.py`](../tests/test_neural.py).
