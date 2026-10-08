# 29. Learning values

Knowing the entire transition model makes planning straightforward. Usually an agent only observes what happened after its own actions. Value learning updates predictions from that experience. This lesson connects a single hand-computable update to a small learned policy with an exact reference solution.

**Prerequisites:** discounted returns and the two-state robot in lesson 28; arithmetic with learning rates. **Objectives:** distinguish Monte Carlo and temporal-difference targets, explain Q-learning versus SARSA, handle episode boundaries correctly, and identify the selection/evaluation split in Double DQN.

## What an update means

Monte Carlo uses a completed observed return. One-step temporal-difference learning uses a reward plus a prediction of continuation:

```text
target = reward + gamma * V(next_state)     # if task continues
delta = target - V(state)
V(state) <- V(state) + alpha * delta
```

A current value of 3, reward of 2, next value of 4, discount 0.9, and step size 0.25 give target 5.6, error 2.6, and updated value 3.65. The target is evidence constructed partly from another estimate; it is not an observed truth.

For control, Q-learning substitutes the largest next-state action value. SARSA substitutes the value of the actual next action selected by the behavior policy. With current Q = 2, reward = -1, gamma = 0.95, alpha = 0.2, next values `[5, 1]`, and exploratory next action valued at 1:

```text
Q-learning target = -1 + .95*5 = 3.75; new Q = 2.35
SARSA target       = -1 + .95*1 = -.05; new Q = 1.59
```

Q-learning targets greedy future behavior even while exploratory actions collect data. SARSA evaluates the continuing behavior policy. Neither is universally safer. A legal-action mask must apply consistently when collecting experience and computing a maximum.

True termination removes the continuation value. Administrative truncation ends a collection segment while the task continues, so it retains bootstrapping from the actual final observation. Never substitute the observation from an automatically reset environment. An actual finite task horizon, by contrast, belongs in the task definition and often in its state.

## Run and interpret

```bash
python -m mlfirst --lesson 29 --seed 42
```

`learned_q` should favor recharging when tired. `max_value_error` compares the learned best values with exact dynamic programming; it should be very small in this deterministic example. `visits` confirms which state-action pairs received evidence. The three update fields reproduce the arithmetic above, while `terminal_target` is -3 and `truncated_target` is 87.

```python
from mlfirst.decisions import td_target, double_q_target
print(td_target(-3, 100, terminated=True))
print(td_target(-3, 100, truncated=True))
print(double_q_target(0, [5, 7], [4, 3], discount=1))
```

The last result is 3: online estimates select the second action, and target estimates evaluate that same action. Ordinary DQN would take the target-network maximum, 4. This addresses a dependence introduced by maximizing noisy estimates; it does not establish that either estimate is correct.

The runner reuses [the book's tabular learner](../book_code/companion_rl.py) with your requested seed. The original printed experiment uses seed 17. It has fixed learning rate 0.1 and deterministic transitions. A small observed error here is not a general convergence guarantee for that schedule, neural approximation, or offline datasets.

## Exercises

1. Compute Expected SARSA's target when the next action probabilities are `[0.9, 0.1]` for values `[5,1]`. Hint: average before discounting.
2. What happens if a logging time limit is incorrectly marked terminal? Hint: compare -3 and 87.
3. Reverse the online values to `[8,7]` in the Double DQN call. Hint: selection changes; target estimates stay fixed.

<details><summary>Worked solutions</summary>

1. Expected continuation is `0.9*5 + 0.1*1 = 4.6`. The target is `-1 + .95*4.6 = 3.37`, and the update from 2 with alpha .2 is 2.274.
2. Future value is artificially discarded, creating a downward bias near collection boundaries. The specific target changes by 90.
3. Online values now select action zero. Target evaluation returns 4; no target-network maximum is needed.

</details>

Neural DQN, replay training, and full offline RL are outside this small executable lesson. For the conceptual foundation, consult [Sutton and Barto](http://incompleteideas.net/book/the-book-2nd.html) and the original book's references.
