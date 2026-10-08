# 25. Generative adversarial models

**Prerequisites:** Gaussian samples, sigmoid classification, binary cross-entropy, and derivatives. **Objective:** train two opposing scalar models, verify each player's gradients, and interpret generation using distributional checks rather than one decreasing loss.

A generative adversarial model couples a generator to a discriminator. The generator transforms random noise into samples. The discriminator learns to distinguish real from generated observations. Their objectives interact, so training is a game whose effective target changes as the other player changes.

This experiment keeps the model deliberately constrained. Real data follow `Normal(2,1)` and noise follows `Normal(0,1)`. The generator is `G(z)=z+mu`; only its location `mu` is learned. The discriminator is `D(x)=sigmoid(a*x+b)`, with learned slope and intercept. Arrays of real and noise samples have shape `[batch]`; the three trainable parameters are scalars. The generator's variance is fixed to one by construction, so it cannot learn an arbitrary distribution or several modes.

The discriminator minimizes

`L_D = mean(-log D(real) - log(1-D(G(z))))`.

The generator uses the non-saturating loss

`L_G = mean(-log D(G(z)))`.

Stable `logaddexp` expressions evaluate these losses without explicitly taking logs of rounded sigmoid probabilities. Holding the discriminator fixed, the generator derivative is `mean((D(fake)-1)*a)`. The discriminator gradients receive contributions from both real and fake examples. The test suite checks each player's derivative by finite differences while holding the other player's parameters fixed.

A hand calculation explains equilibrium values. If real and generated distributions match and the discriminator outputs `.5` everywhere, its loss is `2*log(2)≈1.38629`, while the generator loss is `log(2)≈.693147`. These nonzero losses are compatible with success. They are not thresholds proving that distributions match: an underpowered or poorly optimized discriminator can also be uncertain.

Run:

```bash
python -m mlfirst --lesson 25
```

The loop alternates one discriminator update and one generator update on fresh batches. Inspect `initial_generator_mean`, `learned_generator_mean`, `generated_sample_mean`, `generated_sample_std`, `discriminator_loss`, and `generator_loss`. With seed 42, the location moves from `-1` to about `2.018`, and a fresh sample has mean about `2.004` and standard deviation about `1.011`. Sampling noise explains small deviations from the population values.

A common pitfall is treating a falling generator loss as universal progress; it may instead reflect a weakening discriminator. Another is judging only attractive individual outputs while missing absent modes. This location-only family cannot demonstrate multimodal coverage or image fidelity. Its value is that both the target distribution and the model's limitations are explicit, making the gradient game interpretable.

## Exercises

1. What is the generator gradient when discriminator slope `a=0`?
2. Could this generator fit `Normal(2,4)` where 4 is variance?
3. Why evaluate newly sampled noise after training?

<details><summary>Hints and worked solutions</summary>

1. The derivative is zero regardless of the discriminator intercept; the discriminator supplies no direction in input space.
2. No. Adding a constant changes mean but leaves variance equal to one. A learned scale would be required even for this simple extension.
3. Fixed latent draws show how particular outputs evolve, but fresh draws test behavior across the generator's sampling distribution and avoid selecting only convenient examples.

</details>

Source: [`gan_losses_grad` and `lesson_25`](../mlfirst/neural.py). Verification: [`test_neural.py`](../tests/test_neural.py).
