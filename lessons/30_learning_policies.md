# 30. Learning policies

Instead of learning action values and maximizing them, we can adjust the rule that samples actions. Policy gradients reward probability changes associated with good observed returns. This is useful when action spaces are continuous, although the smallest possible experiment needs only two actions.

**Prerequisites:** derivatives, expectations, discounted returns, and lesson 29's temporal-difference error. **Objectives:** derive a sigmoid policy gradient, understand an action-independent baseline, compute generalized advantage estimation, and handle both signs in PPO's clipped objective.

## From probabilities to gradients

Let action A occur with probability `p = sigmoid(theta)` and B with probability `1-p`. If their rewards are 4 and 1, expected reward is `J(theta) = 1 + 3p`. Therefore

```text
dJ/dtheta = 3p(1-p)
score(A) = d log p / dtheta = 1-p
score(B) = d log(1-p) / dtheta = -p
sample_gradient = score(sampled_action) * (reward - baseline)
```

At p = 0.25, an A sample with reward 4 and zero baseline contributes 3. That is one stochastic estimate, not the exact gradient. Averaging over both actions gives `3*.25*.75 = .5625`. An action-independent baseline has zero expected score-weighted contribution; it can reduce variance without changing the intended gradient. Treat the baseline as fixed during the actor's update.

For several steps, a critic supplies residuals `delta_t = reward + gamma*V(next) - V(current)`. Generalized advantage estimation sums residuals with factor `gamma*lambda`. For `[1,-0.5,2]`, gamma .9 and lambda .8, the first advantage is

```text
1 + .72*(-.5) + .72^2*2 = 1.6768
```

The implementation separates two masks: termination removes bootstrapping, while termination or truncation stops the advantage trace from entering a reset episode. At lambda one, a completed episode telescopes to return minus its initial value.

## PPO's sign-sensitive minimum

Using the same sampled action's new and saved old probabilities, define ratio `r = new_probability / old_probability`. With a fixed advantage A, PPO maximizes

```text
min(r*A, clip(r, 1-epsilon, 1+epsilon)*A)
```

For epsilon .2, `(r,A)=(1.5,2)` gives 2.4. `(0.5,-2)` gives -1.6. But `(1.4,-3)` gives -4.2, not -3.6: the unfavorable probability increase still receives its full penalty. Clipping alone implements a different objective. This surrogate does not impose a hard bound on every probability or on policy divergence.

## Run and interpret

```bash
python -m mlfirst --lesson 30 --seed 42
```

`analytic_gradient` and `finite_difference_gradient` should agree. `ppo_clipped_objectives` reports the three examples. `gae_example[0]` is 1.6768. The separate `seeded_reinforce_extension` samples batches of bandit actions, performs ascent, and reports increased probability of A and expected reward.

```python
from mlfirst.decisions import gae, clipped_surrogate, train_bandit
print(clipped_surrogate([1.5, .5, 1.4], [2, -2, -3]))
print(train_bandit(seed=7))
```

The book supplied the gradient-check and clipping examples in [companion_rl.py](../book_code/companion_rl.py). The sampled bandit trainer and boundary-aware GAE are repository additions. The runnable trainer is REINFORCE, while the PPO function is an objective kernel; this repository does not present that bandit as a full PPO, A2C, or DDPG implementation.

## Exercises

1. Set rewards A and B equal. What is the exact gradient? Hint: differentiate expected reward.
2. Compute the first advantage when lambda is zero. Hint: later residual weights disappear.
3. For ratio .5 and advantage +2, is clipping active in the final objective? Hint: take the minimum, not just the clipped product.

<details><summary>Worked solutions</summary>

1. Expected reward is constant in theta, so the gradient is zero. Individual random samples can still have nonzero gradient contributions.
2. The advantage is the first residual, 1. This relies completely on the one-step critic target.
3. The raw term is 1 and the clipped term is 1.6. The objective is 1, retaining the penalty for decreasing a beneficial action's probability.

</details>

Further reading: [the original PPO paper](https://arxiv.org/abs/1707.06347) and [generalized advantage estimation](https://arxiv.org/abs/1506.02438). Full continuous-control implementations also need correctly transformed action densities, exploration, and independent policy evaluation.
