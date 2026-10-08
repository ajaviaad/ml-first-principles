# 26. Diffusion and score based generation

**Prerequisites:** Gaussian means and variances, conditional distributions, and array operations. **Objective:** implement a forward corruption and a genuine reverse sampling chain while clearly identifying what is known analytically. This lesson uses an **oracle Gaussian population model**, not a trained denoising network.

A forward diffusion step shrinks the previous observation and adds Gaussian noise:

`x_t = sqrt(alpha_t)*x_(t-1) + sqrt(1-alpha_t)*epsilon`.

Let `alpha_bar_t` be the product of the alphas up to step `t`. One can draw a noisy example directly using

`x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*epsilon`.

For `x_0=.8`, `alpha_bar=.64`, and `epsilon=-.5`, the result is `.8×.8 + .6×(-.5)=.34`. If that exact noise draw is known, algebra recovers `.8`. A real denoiser normally does not know the actual draw or the original observation; ambiguity remains at high noise levels.

Our reverse sampler instead knows the population distribution `x_0~Normal(m,v)`, with `m=2,v=.25`. At the previous noise level, write `m_p=sqrt(alpha_bar_previous)*m` and `v_p=alpha_bar_previous*v+1-alpha_bar_previous`. After another corruption step, the current mean and variance are `m_c=sqrt(alpha)*m_p` and `v_c=alpha*v_p+1-alpha`. Gaussian conditioning gives

`E[x_previous | x_current] = m_p + sqrt(alpha)*v_p/v_c*(x_current-m_c)`

`Var[x_previous | x_current] = v_p - alpha*v_p²/v_c`.

The function samples from this exact conditional by adding Gaussian noise with that variance. It does not retrieve an individual training point. Both inputs and outputs are vectors of independent scalar samples, shape `[sample_count]`. Sixty reverse steps transform 12,000 draws from the **exact terminal marginal** into draws from the target population. Starting at a standard Gaussian instead would introduce a finite-schedule approximation.

Run:

```bash
python -m mlfirst --lesson 26
```

Inspect `forward_example`, `recovered_with_known_noise`, `terminal_alpha_bar`, `reverse_sample_mean`, and `reverse_sample_std`. Seed 42 produces final mean about `1.9983` and standard deviation about `.4968`, near the target 2 and `.5`. The test suite independently verifies reverse-step moments. No loss or trained denoiser is reported because none is fitted.

The key limitation is deliberate: knowing the data mean and variance solves this Gaussian family analytically. Real image distributions require learned, noise-level-dependent structure. More sampling steps cannot recover information a conditioning input never supplied. Similarly, deterministic conditional means alone would remove part of the conditional variance and would generally sample the wrong distribution.

## Exercises

1. What does the forward sample become when `alpha_bar=1`?
2. Find the marginal variance when `alpha_bar=.4` and data variance is `.25`.
3. Explain why exact population reversal does not identify one unique clean ancestor.

<details><summary>Hints and worked solutions</summary>

1. The noise coefficient vanishes and the result is the clean observation.
2. It is `.4×.25 + .6 = .7`; signal variance and corruption variance add because the variables are independent.
3. Several clean values can plausibly produce the same noisy value. The reverse conditional describes their distribution and retains positive uncertainty. Recovering a population is different from inverting a particular hidden random draw.

</details>

Source: [`diffusion_forward`, `gaussian_reverse_step`, and `lesson_26`](../mlfirst/neural.py). Verification: [`test_neural.py`](../tests/test_neural.py).
