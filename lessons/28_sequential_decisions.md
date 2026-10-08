# 28. Sequential decisions

A supervised predictor answers a question about an observation. A decision-making agent also changes what it will observe next. Imagine a delivery robot: working brings income but drains its battery; recharging earns nothing immediately but enables another profitable job. This lesson makes that tradeoff explicit and solves it completely.

**Prerequisites:** expectations, geometric series, dictionaries, and basic NumPy. **Objectives:** describe a Markov decision process, compute discounted returns, distinguish a policy from a value function, and verify a Bellman solution using algebra. You can work through this lesson without the book.

## The model and the calculation

A state must include information needed to predict the next outcome given an action. Position alone is insufficient when battery level changes what is possible. The Markov assumption is about this state representation. A policy says which action to take; a value predicts the return obtained by following a policy.

For rewards following time t, the discounted return is

```text
G_t = R_(t+1) + gamma R_(t+2) + gamma^2 R_(t+3) + ...
Q(s,a) = sum_outcomes p(outcome) [reward + gamma V(next_state)]
V_new(s) = max over legal actions a of Q(s,a)
```

The continuation term is zero after genuine termination. For a continuing finite model, `0 <= gamma < 1` makes the Bellman optimality operator a contraction. Synchronous value iteration computes every new state value from the same previous table. The stopping threshold measures successive-update size; the remaining value error can be larger by a factor involving `1/(1-gamma)`.

The book's robot has two states. Working when ready earns 4 and leaves it tired. Gentle work when tired earns 1 and leaves it tired. Recharging earns 0 and restores readiness. With gamma 0.9, the proposed work/recharge cycle obeys

```text
V_ready = 4 + 0.9 V_tired
V_tired = 0.9 V_ready
V_ready = 4 / (1 - 0.9^2) = 21.05263158
V_tired = 18.94736842
```

Check the competing action: one gentle task followed by that cycle is worth `1 + 0.9*18.94736842 = 18.05263158`. Recharging is better. This comparison verifies that the proposed policy is optimal, rather than merely evaluating it.

## Run and inspect

From the repository root after installation:

```bash
python -m mlfirst --lesson 28
```

The result contains `book_robot_values`, `optimal_policy`, and analytic values. Expect `work` in ready and `recharge` in tired, with numerical values matching the hand calculation. The two delayed-job fields show `-1 + 0.9*10 = 8` and `-1 + 0.5*10 = 4`; an immediate reward of 6 wins only in the second comparison. Changing discount changes the objective.

Use the reusable functions directly:

```python
from mlfirst.decisions import value_iteration, discounted_returns
print(value_iteration(discount=0.5))
print(discounted_returns([-1, 10], discount=0.9))
```

`value_iteration` validates probabilities and legal successor states, then calls the original implementation in [companion_rl.py](../book_code/companion_rl.py). These calculations reproduce the book's model. The CLI seed is recorded but no randomness is needed.

## Exercises

1. Find the discount at which gentle work and recharging tie in the tired state. Hint: compare the perpetual gentle-work value with the cycle.
2. Replace the reward 4 by 2. Would the original discount still favor recharging? Hint: repeat the action-value comparison.
3. Explain why a robot with ten seconds remaining may require another state variable. Hint: a worthwhile recharge depends on future opportunities.

<details><summary>Worked solutions</summary>

1. The cycle gives tired value `4*gamma/(1-gamma^2)`; gentle work gives `1/(1-gamma)`. Equality gives `4*gamma = 1+gamma`, hence gamma = 1/3.
2. The cycle gives ready value `2/.19 = 10.5263` and tired value `9.4737`. Gentle work forever gives 10, so recharging is no longer optimal. Always gentle when tired gives ready value 11.
3. Include remaining time or use time-indexed values. A stationary battery-only model aliases different planning horizons.

</details>

The example has exact transitions, tiny state space, and an explicit reward. It does not validate a learned simulator, solve partial observability, or assess physical safety. Reference: [Sutton and Barto's reinforcement-learning textbook](http://incompleteideas.net/book/the-book-2nd.html).
