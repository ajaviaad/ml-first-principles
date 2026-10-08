# companion_rl.py
"""Small, auditable experiments for chapters 28-30.

Requires Python 3.11+; uses only the standard library.
Run with: python3 companion_rl.py

The environment is the constructed two-state robot from chapter 28.
This is an instructional experiment, not a physical robot controller.
"""

from dataclasses import dataclass
from math import exp, isclose
from random import Random


@dataclass(frozen=True)
class Outcome:
    probability: float
    reward: float
    next_state: str
    terminated: bool = False


MODEL = {
    "ready": {"work": (Outcome(1.0, 4.0, "tired"),)},
    "tired": {
        "gentle": (Outcome(1.0, 1.0, "tired"),),
        "recharge": (Outcome(1.0, 0.0, "ready"),),
    },
}


def value_iteration(model, discount=0.9, tolerance=1e-12):
    """Synchronous Bellman optimality updates for a finite known model."""
    if not 0 <= discount < 1:
        raise ValueError("This implementation requires 0 <= discount < 1.")
    values = {state: 0.0 for state in model}
    for _ in range(100_000):
        action_values = {
            state: {
                action: sum(
                    out.probability
                    * (
                        out.reward
                        + discount
                        * (0.0 if out.terminated else values[out.next_state])
                    )
                    for out in outcomes
                )
                for action, outcomes in actions.items()
            }
            for state, actions in model.items()
        }
        updated = {
            state: max(action_values[state].values()) for state in model
        }
        residual = max(abs(updated[s] - values[s]) for s in model)
        values = updated
        if residual < tolerance:
            policy = {
                state: max(action_values[state], key=action_values[state].get)
                for state in model
            }
            return values, policy
    raise RuntimeError(
        "Value iteration did not reach the requested tolerance."
    )


def q_target(reward, next_values, discount, *, terminated):
    """A collection-window truncation should pass terminated=False."""
    return reward if terminated else reward + discount * max(next_values)


def q_learning(steps=30_000, seed=17, epsilon=0.2, discount=0.9):
    "Learn from sampled interactions, without consulting other transitions."
    rng = Random(seed)
    q = {s: {a: 0.0 for a in actions} for s, actions in MODEL.items()}
    visits = {(s, a): 0 for s, actions in MODEL.items() for a in actions}
    state = "ready"
    for _ in range(steps):
        actions = list(q[state])
        if rng.random() < epsilon:
            action = rng.choice(actions)
        else:
            best = max(q[state].values())
            action = rng.choice([a for a in actions if q[state][a] == best])
        # All transitions in this constructed model are deterministic.
        outcome = MODEL[state][action][0]
        visits[state, action] += 1
        target = q_target(
            outcome.reward,
            q[outcome.next_state].values(),
            discount,
            terminated=outcome.terminated,
        )
        q[state][action] += 0.1 * (target - q[state][action])
        state = outcome.next_state
    return q, visits


def clipped_surrogate(ratio, advantage, epsilon=0.2):
    clipped = min(max(ratio, 1.0 - epsilon), 1.0 + epsilon)
    return min(ratio * advantage, clipped * advantage)


def verify():
    values, policy = value_iteration(MODEL)
    analytical_ready = 4.0 / (1.0 - 0.9**2)
    analytical_tired = 0.9 * analytical_ready
    assert isclose(values["ready"], analytical_ready, abs_tol=1e-9)
    assert isclose(values["tired"], analytical_tired, abs_tol=1e-9)
    assert policy == {"ready": "work", "tired": "recharge"}

    q, visits = q_learning()
    assert all(count > 0 for count in visits.values())
    assert max(abs(max(q[s].values()) - values[s]) for s in values) < 1e-8

    # The same final observation gives different targets when an episode
    # truly ends and when only a collection window ends.
    assert q_target(-3, [100], 0.9, terminated=True) == -3
    assert q_target(-3, [100], 0.9, terminated=False) == 87

    # Chapter 29 uses the same transition but distinct continuation rules.
    q_update = 2 + 0.2 * (-1 + 0.95 * 5 - 2)
    sarsa_update = 2 + 0.2 * (-1 + 0.95 * 1 - 2)
    assert isclose(q_update, 2.35)
    assert isclose(sarsa_update, 1.59)

    assert isclose(clipped_surrogate(1.5, 2), 2.4)
    assert isclose(clipped_surrogate(0.5, -2), -1.6)
    assert isclose(clipped_surrogate(1.4, -3), -4.2)

    # Independently verify the expected one-step policy gradient by a
    # finite difference of exact expected reward, including both actions.
    theta = -0.7
    reward_a, reward_b = 4.0, 1.0
    probability = 1.0 / (1.0 + exp(-theta))
    gradient = (
        probability * (1 - probability) * reward_a
        + (1 - probability) * (-probability) * reward_b
    )

    def objective(parameter):
        p = 1.0 / (1.0 + exp(-parameter))
        return p * reward_a + (1.0 - p) * reward_b

    h = 1e-5
    finite_difference = (
        objective(theta + h) - objective(theta - h)
    ) / (2 * h)
    assert isclose(gradient, finite_difference, rel_tol=1e-8)
    return values, policy, q, visits


if __name__ == "__main__":
    values, policy, q, visits = verify()
    print("All numerical checks passed.")
    for state in MODEL:
        print(
            f"{state}: exact value={values[state]:.6f}, "
            f"action={policy[state]}"
        )
        for action, estimate in q[state].items():
            print(
                f"  {action}: learned Q={estimate:.6f}, "
                f"visits={visits[state, action]}"
            )
# End of companion_rl.py
