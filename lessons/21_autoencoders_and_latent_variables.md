# 21. Autoencoders and latent variables

**Prerequisites:** squared error, matrix multiplication, Gaussian distributions, and gradients. **Objective:** compress correlated observations, train a tiny probabilistic encoder/decoder, and separate reconstruction quality from the meaning of a latent coordinate.

An autoencoder maps an observation `x` into a code `z`, then decodes `z` into a reconstruction. A narrow code creates a bottleneck. For centered two-dimensional data near a line, an optimal one-dimensional linear squared-error autoencoder can be obtained from singular value decomposition. If `v[1,2]` is the first right singular vector, the code is `X_centered @ v.T` and reconstruction is `code @ v + center`. SVD fits the subspace exactly here; this first experiment does not use iterative neural training.

A variational autoencoder additionally specifies a latent prior and approximate posterior. Our second experiment uses one latent variable, `q(z|x)=Normal(mu(x),sigma²)`, with linear mean `mu=X @ encoder + bias` and one shared log variance. The decoder is linear, and its observation variance is fixed to one. A standard normal prior gives

`KL = .5 * (mu² + sigma² - 1 - log(sigma²))`.

For `mu=1` and `sigma=.5`, this is approximately `.818147`. Reparameterization writes `z=mu+sigma*epsilon`, with independent standard normal `epsilon`. If `epsilon=-2`, the sample is zero. Holding that draw fixed, the derivative of `z` with respect to `mu` is 1 and with respect to `sigma` is -2.

The trained objective is mean half-squared reconstruction error plus mean KL. It is a Monte Carlo negative evidence lower bound, omitting the fixed Gaussian observation constant. The implementation uses coefficient `beta=1`; choosing a different coefficient changes the objective away from the ordinary negative ELBO. Shapes are `X[n,2]`, encoder `[2,1]`, latent `[n,1]`, and decoder `[1,2]`.

Run:

```bash
python -m mlfirst --lesson 21
```

Compare `linear_autoencoder_mse` with `mean_only_mse`. Inspect the two `vae_*negative_elbo_without_constant` values, `vae_posterior_std`, `gaussian_kl_example`, and `reparameterized_sample`. Seed 42 gives linear reconstruction MSE about `.0035`, against `1.53` for predicting the mean. The VAE objective falls from about `1.55` to `1.02`. Training draws fresh noise; evaluation uses one fixed draw for an interpretable comparison.

Reconstruction is not a guarantee that a representation preserves labels, anomalies, or meaningful human concepts. A diagonal posterior can miss multimodal explanations. A small KL can mean good regularization or an ignored latent code; inspect reconstruction and sensitivity to latent changes together. The gradients are checked with fixed noise, because resampling during finite differences would mix sampling variation with derivative error.

## Exercises

1. Evaluate the KL for `mu=0,sigma=1`.
2. What happens to KL as positive `sigma` approaches zero with fixed mean?
3. Why center the data before the SVD autoencoder?

<details><summary>Hints and worked solutions</summary>

1. Each term cancels, yielding zero: the posterior matches the prior.
2. `-log(sigma²)` grows without bound. A nearly deterministic posterior pays for its concentration.
3. Centering allows the one-dimensional subspace to describe variation around the learned mean. Without centering, a large offset can dominate the first singular direction and waste the limited code capacity.

</details>

Source: [`vae_loss_grad` and `lesson_21`](../mlfirst/neural.py), [`diagonal_gaussian_kl`](../book_code/companion_deep.py). Verification: [`test_neural.py`](../tests/test_neural.py).
