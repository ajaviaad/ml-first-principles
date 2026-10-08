# 24. Language models

**Prerequisites:** categorical probability, logarithms, and counting. **Objective:** fit a word bigram model, calculate heldout perplexity, and sample a continuation. The mechanism is intentionally small enough to inspect without a neural framework or external text corpus.

A language model assigns probabilities to token sequences. The chain rule writes their probability as a product of conditional probabilities. A bigram model approximates each conditional using only the preceding token:

`P(w1,...,wT | w0) = product_t P(wt | w_(t-1))`.

This is a severe context limitation. It can learn that “the” tends to precede colors, but cannot retain a subject or instruction across a paragraph. A transformer language model uses a much richer context-dependent parameterization; the probabilistic evaluation idea remains related.

The lesson declares nine tokens, including beginning and ending markers, before fitting. Four tiny training sentences contribute transition counts. Each row receives positive additive smoothing:

`P(next=j | current=i) = (count[i,j] + a) / (sum_j count[i,j] + a*V)`.

Here `V` is vocabulary size and `a=.5`. The transition table has shape `[V,V]`, with each row summing to one. Smoothing assigns nonzero probability to transitions absent from the training text. This toy implementation allows every vocabulary item, including special markers, as an output from every row; unusual generated sequences are therefore possible and informative.

For a hand example with vocabulary `[a,b]`, one observed sequence `[a,b,a]`, and smoothing 1, both observed cross-token transitions have probability `2/3`; both self-transitions have probability `1/3`. Perplexity for observed probabilities `p_t` is

`exp(-mean(log(p_t)))`.

For `[.5,.5,.25,.5]`, it equals `32**.25≈2.378414`. A uniform model over nine tokens has perplexity 9. This comparison has meaning only when tokenization and evaluated token positions match.

Run:

```bash
python -m mlfirst --lesson 24
```

Inspect `heldout_perplexity`, `uniform_perplexity`, `hand_perplexity`, `row_sum_error`, and `generated_tokens`. Evaluation uses two new sentence compositions that were not counted during fitting. Seed 42 gives heldout perplexity about `3.157`; sampling may produce repeated or awkward tokens. Samples use the previous sampled token as the next context and stop at the ending marker or after twenty steps.

Low perplexity on this tiny controlled text does not establish factual reliability, instruction following, or general language ability. Changing the vocabulary changes the metric's scale. Training/test duplicates make evaluation optimistic. Sampling randomness also differs from model uncertainty: a model can sample varied outputs while assigning poor probabilities to the actual task distribution.

## Exercises

1. Find perplexity if every observed token has probability `.25`.
2. Explain why smoothing is useful when a heldout transition was unseen.
3. Compare greedy decoding with the sampling used here.

<details><summary>Hints and worked solutions</summary>

1. Every negative log probability is `log(4)`, so perplexity is 4 regardless of sequence length.
2. Without smoothing, the transition may have probability zero and infinite negative log-likelihood. Positive smoothing gives a finite evaluation while distributing some mass away from observed transitions.
3. Greedy decoding repeatedly selects the largest probability. Sampling draws from the full categorical distribution. Greedy output is repeatable for fixed tie-breaking but is not guaranteed to maximize the entire sequence probability.

</details>

Source: [`fit_bigram`, `perplexity`, and `lesson_24`](../mlfirst/neural.py). Verification: [`test_neural.py`](../tests/test_neural.py).
