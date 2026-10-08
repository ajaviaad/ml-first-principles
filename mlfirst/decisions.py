"""Chapters 28–32: decisions, monitoring and a complete regression investigation.

The original book programs remain in ``book_code``. These small reusable tools
add validation, explicit data boundaries and exercises; they are not production
robot controllers or a replacement for a full RL training library.
"""
from dataclasses import dataclass
from math import ceil
import numpy as np
from book_code import companion_rl as book_rl
from book_code import companion_investigation as book_investigation


def _vector(values, name):
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite one-dimensional array")
    return result


def _discount(discount):
    if not np.isfinite(discount) or not 0 <= discount <= 1:
        raise ValueError("discount must lie in [0, 1]")


def value_iteration(model=book_rl.MODEL, discount=0.9, tolerance=1e-12):
    """Validate a finite MDP, then run the book's synchronous Bellman solver.

    Each state maps legal action names to sequences of book_rl.Outcome objects.
    Terminal outcomes need no modeled successor. Discount must be below one.
    The residual tolerance is an update tolerance, not a direct value-error bound.
    """
    if not model or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("a nonempty model and positive finite tolerance are required")
    _discount(discount)
    if discount == 1:
        raise ValueError("discount must be below one for this infinite-horizon solver")
    for actions in model.values():
        if not actions:
            raise ValueError("each modeled state needs at least one legal action")
        for outcomes in actions.values():
            if not outcomes:
                raise ValueError("each action needs outcomes")
            probabilities = np.array([o.probability for o in outcomes])
            if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
                raise ValueError("transition probabilities must be finite and nonnegative")
            if not np.isclose(probabilities.sum(), 1.0, atol=1e-12, rtol=0):
                raise ValueError("transition probabilities must sum to one")
            for outcome in outcomes:
                if not np.isfinite(outcome.reward):
                    raise ValueError("rewards must be finite")
                if not outcome.terminated and outcome.next_state not in model:
                    raise ValueError("nonterminal successor is absent from the model")
    return book_rl.value_iteration(model, discount, tolerance)


def discounted_returns(rewards, discount=0.9, bootstrap=0.0):
    """Backward returns for ONE segment; bootstrap is zero after true termination."""
    rewards = _vector(rewards, "rewards")
    _discount(discount)
    if not np.isfinite(bootstrap):
        raise ValueError("bootstrap must be finite")
    result = np.empty_like(rewards)
    running = float(bootstrap)
    for index in range(len(rewards) - 1, -1, -1):
        running = rewards[index] + discount * running
        result[index] = running
    return result


def td_target(reward, next_value, discount=0.9, *, terminated=False, truncated=False):
    """One-step target; administrative truncation retains the final observation's value.

    The next_value must describe the observation before any environment reset.
    If both boundary flags are true, genuine termination takes precedence.
    """
    _discount(discount)
    if not np.isfinite([reward, next_value]).all():
        raise ValueError("reward and next_value must be finite")
    return float(reward if terminated else reward + discount * next_value)


def double_q_target(reward, online_values, target_values, discount=0.9, *, terminated=False):
    """Select with online values, evaluate with target values (legal actions only)."""
    online = _vector(online_values, "online_values")
    target = _vector(target_values, "target_values")
    if online.shape != target.shape:
        raise ValueError("online and target values must have matching shapes")
    return td_target(reward, target[np.argmax(online)], discount, terminated=terminated)


def gae(rewards, values, next_values, terminated, truncated, discount=0.9, lam=0.8):
    """GAE with separate bootstrap and trace masks, including concatenated episodes.

    Arrays contain one entry per transition. next_values uses the actual final
    observation before reset. Termination suppresses bootstrapping; BOTH boundary
    types stop traces so residuals from the reset episode cannot leak backward.
    """
    rewards, values, next_values = [_vector(x, n) for x, n in
                                  ((rewards, "rewards"), (values, "values"),
                                   (next_values, "next_values"))]
    terminal, cut = np.asarray(terminated, bool), np.asarray(truncated, bool)
    if any(x.shape != rewards.shape for x in (values, next_values, terminal, cut)):
        raise ValueError("all transition arrays must have identical shapes")
    _discount(discount)
    if not np.isfinite(lam) or not 0 <= lam <= 1:
        raise ValueError("lam must lie in [0, 1]")
    delta = rewards + discount * next_values * (~terminal) - values
    advantages = np.empty_like(delta)
    continuation = 0.0
    for i in range(len(delta) - 1, -1, -1):
        continuation = delta[i] + discount * lam * (not (terminal[i] or cut[i])) * continuation
        advantages[i] = continuation
    return advantages


def clipped_surrogate(ratios, advantages, epsilon=0.2):
    """Per-sample PPO objectives to maximize; use matching saved old-action densities."""
    ratios = _vector(ratios, "ratios")
    advantages = _vector(advantages, "advantages")
    if ratios.shape != advantages.shape or np.any(ratios < 0):
        raise ValueError("matching shapes and nonnegative ratios are required")
    if not np.isfinite(epsilon) or not 0 <= epsilon < 1:
        raise ValueError("epsilon must lie in [0, 1)")
    return np.minimum(ratios * advantages,
                      np.clip(ratios, 1 - epsilon, 1 + epsilon) * advantages)


def sigmoid(theta):
    """Stable scalar sigmoid, including large negative inputs."""
    if not np.isfinite(theta):
        raise ValueError("theta must be finite")
    return float(1 / (1 + np.exp(-theta)) if theta >= 0 else
                 np.exp(theta) / (1 + np.exp(theta)))


def policy_gradient(theta, reward_a=4.0, reward_b=1.0):
    """Exact gradient of the expected reward of a two-action sigmoid policy."""
    if not np.isfinite([reward_a, reward_b]).all():
        raise ValueError("rewards must be finite")
    p = sigmoid(theta)
    return p * (1 - p) * (reward_a - reward_b)


def train_bandit(seed=42, updates=200, batch_size=128, learning_rate=0.1):
    """Sampled REINFORCE on a one-step, two-action world; an exact baseline is fixed per batch."""
    if updates < 1 or batch_size < 1 or learning_rate <= 0 or not np.isfinite(learning_rate):
        raise ValueError("updates, batch_size and learning_rate must be positive")
    rng = np.random.default_rng(seed)
    theta = -0.7
    initial = sigmoid(theta)
    for _ in range(updates):
        p = sigmoid(theta)
        actions = rng.binomial(1, p, batch_size)
        rewards = 1 + 3 * actions
        baseline = 1 + 3 * p
        theta += learning_rate * float(np.mean((actions - p) * (rewards - baseline)))
    return {"initial_probability_a": initial, "final_probability_a": sigmoid(theta),
            "initial_expected_reward": 1 + 3 * initial,
            "final_expected_reward": 1 + 3 * sigmoid(theta), "updates": updates}


def drift_report(reference, current):
    """Per-feature mean shifts in reference standard deviations plus missingness.

    NaNs mean missing. Infinities and all-missing reference columns are rejected.
    A constant reference uses scale one and is explicitly flagged. An entirely
    missing current column returns None for its mean shift. No hypothesis-test
    threshold or predictive-damage claim is implied by these descriptive metrics.
    """
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    if (reference.ndim != 2 or current.ndim != 2 or not len(reference) or not len(current)
            or not reference.shape[1] or reference.shape[1] != current.shape[1]):
        raise ValueError("nonempty matrices with matching feature counts are required")
    if np.isinf(reference).any() or np.isinf(current).any() or np.isnan(reference).all(axis=0).any():
        raise ValueError("infinite inputs or an all-missing reference column are invalid")
    means, scales = np.nanmean(reference, axis=0), np.nanstd(reference, axis=0)
    result = []
    for j in range(reference.shape[1]):
        observed = current[:, j][~np.isnan(current[:, j])]
        shift = None if not len(observed) else float((observed.mean() - means[j]) /
                                                    (scales[j] if scales[j] else 1))
        result.append({"feature": j, "standardized_mean_shift": shift,
                       "reference_missing_fraction": float(np.isnan(reference[:, j]).mean()),
                       "current_missing_fraction": float(np.isnan(current[:, j]).mean()),
                       "constant_reference": bool(scales[j] == 0)})
    return result


def _xy(x, y):
    x, y = np.asarray(x, dtype=float), _vector(y, "y")
    if x.ndim != 2 or len(x) != len(y) or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError("x must be a finite matrix with one row per outcome")
    return x, y


def permutation_importance(predict, x, y, repeats=20, seed=42):
    """Held-out MSE increase after column permutation; never fits or changes the input.

    Correlated columns can substitute for each other. Permutations may create
    impossible feature combinations, so these are model diagnostics, not causes.
    """
    x, y = _xy(x, y)
    if not isinstance(repeats, (int, np.integer)) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    def loss(data):
        prediction = _vector(predict(data), "prediction")
        if prediction.shape != y.shape:
            raise ValueError("predict must return one prediction per row")
        return float(np.mean((y - prediction) ** 2))
    baseline, rng = loss(x), np.random.default_rng(seed)
    increases = np.empty((repeats, x.shape[1]))
    for j in range(x.shape[1]):
        for r in range(repeats):
            disrupted = x.copy()
            disrupted[:, j] = rng.permutation(disrupted[:, j])
            increases[r, j] = loss(disrupted) - baseline
    return {"baseline_mse": baseline, "mean_mse_increase": increases.mean(axis=0).tolist(),
            "permutation_std": increases.std(axis=0).tolist(), "repeats": repeats}


def linear_explanation(x, coefficients, intercept=0.0, baseline=None):
    """Exact additive decomposition for a linear predictor relative to a baseline.

    Returns contributions summing to prediction minus baseline_prediction. The
    baseline is a modeling choice. Attributions describe the model, not causality.
    """
    x, coefficients = _vector(x, "x"), _vector(coefficients, "coefficients")
    baseline = np.zeros_like(x) if baseline is None else _vector(baseline, "baseline")
    if x.shape != coefficients.shape or baseline.shape != x.shape or not np.isfinite(intercept):
        raise ValueError("matching feature shapes and a finite intercept are required")
    return {"baseline_prediction": float(intercept + baseline @ coefficients),
            "contributions": ((x - baseline) * coefficients).tolist(),
            "prediction": float(intercept + x @ coefficients)}


def partial_dependence(predict, x, column, grid):
    """Average predictions after replacing one column; inspect plausibility of those rows."""
    x = np.asarray(x, float)
    grid = _vector(grid, "grid")
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all() or not 0 <= column < x.shape[1]:
        raise ValueError("finite data and a valid column are required")
    result = []
    for value in grid:
        changed = x.copy()
        changed[:, column] = value
        prediction = _vector(predict(changed), "prediction")
        if len(prediction) != len(x):
            raise ValueError("predict must return one prediction per row")
        result.append(float(prediction.mean()))
    return result


@dataclass(frozen=True)
class SelectedDemandModel:
    """A fitted selection result; prediction uses the book's fixed feature map."""
    kind: str
    coefficients: np.ndarray
    validation_scores: dict
    constant_coefficients: np.ndarray

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
            raise ValueError("demand features must be finite rows of [t, promotion, weekend]")
        return book_investigation.design(x, self.kind) @ self.coefficients


def fit_select(train, validation):
    """Fit on train and select on validation only; calibration/test are not arguments."""
    train_x, train_y = _xy(*train)
    val_x, val_y = _xy(*validation)
    if train_x.shape[1] != 3 or val_x.shape[1] != 3:
        raise ValueError("the demand example requires three input features")
    models, scores = {}, {}
    for kind in ("constant", "linear", "expanded"):
        models[kind] = np.linalg.lstsq(book_investigation.design(train_x, kind), train_y, rcond=None)[0]
        scores[kind] = book_investigation.rmse(val_y, book_investigation.design(val_x, kind) @ models[kind])
    kind = min(scores, key=scores.get)
    for coef in models.values():
        coef.setflags(write=False)
    return SelectedDemandModel(kind, models[kind], scores, models["constant"])


def conformal_radius(residuals, alpha=0.1):
    """Finite-sample split-conformal absolute-residual radius and one-based rank.

    Residuals must come from a separate calibration sample with a fixed model.
    Returns infinity if the rank exceeds available scores; finite coverage cannot
    be promised then. Exchangeability supports marginal, not subgroup coverage.
    """
    residuals = _vector(residuals, "residuals")
    if np.any(residuals < 0) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("nonnegative residuals and 0 < alpha < 1 are required")
    rank = int(ceil((len(residuals) + 1) * (1 - alpha)))
    radius = float(np.partition(residuals, rank - 1)[rank - 1]) if rank <= len(residuals) else float("inf")
    return radius, rank


def calibrate(model, calibration, alpha=0.1):
    """Determine an interval radius without changing or selecting model parameters."""
    x, y = _xy(*calibration)
    return conformal_radius(np.abs(y - model.predict(x)), alpha)


def evaluate_demand(model, dataset, radius):
    """Point error, marginal coverage and subgroup counts; does not fit or calibrate."""
    x, y = _xy(*dataset)
    if np.isnan(radius) or radius < 0:
        raise ValueError("radius must be nonnegative")
    residual = y - model.predict(x)
    covered = np.abs(residual) <= radius
    groups = {}
    for name, mask in (("weekday", x[:, 2] == 0), ("weekend", x[:, 2] == 1),
                       ("promoted_weekend", (x[:, 1] == 1) & (x[:, 2] == 1))):
        count = int(mask.sum())
        groups[name] = {"count": count, "coverage": float(covered[mask].mean()) if count else None}
    return {"rmse": float(np.sqrt(np.mean(residual ** 2))), "mae": float(np.abs(residual).mean()),
            "coverage": float(covered.mean()), "covered_count": int(covered.sum()),
            "count": len(y), "mean_interval_width": float(2 * radius), "subgroups": groups}


def paired_bootstrap(loss_baseline, loss_candidate, repeats=2000, seed=314159):
    """Percentile CI for mean paired loss improvement, conditional on fixed predictors.

    Positive means the candidate improves on baseline. Resampling pairs preserves
    within-case dependence. This does not include model-refitting uncertainty.
    """
    baseline, candidate = _vector(loss_baseline, "loss_baseline"), _vector(loss_candidate, "loss_candidate")
    if baseline.shape != candidate.shape or repeats < 2:
        raise ValueError("matching losses and at least two repeats are required")
    difference = baseline - candidate
    rng = np.random.default_rng(seed)
    # Stream bootstrap rows, keeping memory independent of number of repeats.
    means = [float(difference[rng.integers(0, len(difference), len(difference))].mean())
             for _ in range(repeats)]
    return {"mean_improvement": float(difference.mean()),
            "interval_95_percent": np.quantile(means, [0.025, 0.975]).tolist(), "repeats": repeats}


def sample_world(rng, n, temperature_shift=0.0, interaction=12.0):
    """Extend the original simulator with independent input/response shift controls.

    Reusing an RNG seed across scenarios gives common random numbers for more
    direct comparisons. Temperature remains standardized against training 18/5.
    """
    if n < 1 or not np.isfinite([temperature_shift, interaction]).all():
        raise ValueError("positive n and finite world parameters are required")
    x, y = book_investigation.sample(rng, n)
    old_t = x[:, 0].copy()
    x[:, 0] += temperature_shift / 5
    y += 7 * (x[:, 0] - old_t) - 2 * (x[:, 0] ** 2 - old_t ** 2)
    y += (interaction - 12) * x[:, 1] * x[:, 2]
    return x, y


def investigation(seed=42):
    """Seed-controlled extension with strictly separated training/selection/calibration/test."""
    rng = np.random.default_rng(seed)
    train = sample_world(rng, 1000)
    validation = sample_world(rng, 250)
    calibration = sample_world(rng, 250)
    test = sample_world(rng, 250)
    model = fit_select(train, validation)
    radius, rank = calibrate(model, calibration)
    baseline = book_investigation.design(test[0], "constant") @ model.constant_coefficients
    bootstrap = paired_bootstrap((test[1] - baseline) ** 2,
                                 (test[1] - model.predict(test[0])) ** 2, seed=seed + 1)
    scenarios = {}
    for name, shift, interaction in (("nominal_fresh", 0, 12), ("input_shift_only", 5, 12),
                                      ("concept_shift_only", 0, 24), ("both_shifts", 5, 24)):
        world = sample_world(np.random.default_rng(seed + 2), 250, shift, interaction)
        scenarios[name] = evaluate_demand(model, world, radius)
    return {"seed": seed, "split_sizes": {"train": 1000, "validation": 250, "calibration": 250, "test": 250},
            "selected": model.kind, "validation_rmse": model.validation_scores,
            "coefficients": model.coefficients.tolist(), "conformal_radius": radius,
            "conformal_rank": rank, "test": evaluate_demand(model, test, radius),
            "paired_bootstrap": bootstrap, "controlled_scenarios": scenarios}


def lesson_28(seed=42):
    values, policy = value_iteration()
    return {"seed": seed, "book_robot_values": values, "optimal_policy": policy,
            "analytic_ready": 4 / (1 - 0.9 ** 2), "analytic_tired": 0.9 * 4 / (1 - 0.9 ** 2),
            "delayed_job_return_discount_09": float(discounted_returns([-1, 10], 0.9)[0]),
            "delayed_job_return_discount_05": float(discounted_returns([-1, 10], 0.5)[0]),
            "immediate_job_return": 6.0, "note": "Exact book model; no randomness is needed."}


def lesson_29(seed=42):
    q, visits = book_rl.q_learning(seed=seed)
    values, _ = value_iteration()
    return {"seed": seed, "learned_q": q,
            "visits": {f"{s}/{a}": count for (s, a), count in visits.items()},
            "max_value_error": max(abs(max(q[s].values()) - values[s]) for s in values),
            "td_updated_value": 3 + 0.25 * (td_target(2, 4) - 3),
            "q_learning_update": 2 + 0.2 * (td_target(-1, 5, 0.95) - 2),
            "sarsa_update": 2 + 0.2 * (td_target(-1, 1, 0.95) - 2),
            "terminal_target": td_target(-3, 100, terminated=True),
            "truncated_target": td_target(-3, 100, truncated=True),
            "dqn_continuation": 4, "double_dqn_continuation": double_q_target(0, [5, 7], [4, 3], 1),
            "note": "Book Q-learning with requested seed; the printed original uses seed 17."}


def lesson_30(seed=42):
    theta, h = -0.7, 1e-5
    gradient_fd = 3 * (sigmoid(theta + h) - sigmoid(theta - h)) / (2 * h)
    return {"seed": seed, "analytic_gradient": policy_gradient(theta),
            "finite_difference_gradient": gradient_fd,
            "ppo_clipped_objectives": clipped_surrogate([1.5, 0.5, 1.4], [2, -2, -3]).tolist(),
            "gae_example": gae([1, -0.5, 2], [0, 0, 0], [0, 0, 0],
                                [False, False, True], [False] * 3).tolist(),
            "seeded_reinforce_extension": train_bandit(seed),
            "note": "PPO clipping kernel and sampled REINFORCE are distinct experiments, not a full PPO agent."}


def lesson_31(seed=42):
    rng = np.random.default_rng(seed)
    reference = rng.normal(size=(500, 3))
    current = rng.normal(size=(300, 3))
    current[:, 0] += 1.5
    current[:30, 1] = np.nan
    evaluation = rng.normal(size=(300, 3))
    coefficients = np.array([3.0, 0.0, -1.0])
    y = evaluation @ coefficients + rng.normal(0, 0.2, len(evaluation))
    predict = lambda x: x @ coefficients
    return {"seed": seed, "input_monitor": drift_report(reference, current),
            "held_out_permutation_importance": permutation_importance(predict, evaluation, y, seed=seed),
            "local_explanation": linear_explanation([1, 2, 3], coefficients, baseline=reference.mean(axis=0)),
            "partial_dependence_feature_0": partial_dependence(predict, evaluation, 0, [-1, 0, 1]),
            "response_plan": {"missingness": "Inspect input source and apply a documented fallback.",
                              "mean_shift": "Inspect population changes; obtain representative labels before judging damage."},
            "note": "New synthetic monitoring exercise; no production thresholds or causal effects are established."}


def lesson_32(seed=42):
    return {"original_book_reproduction": book_investigation.run(),
            "seeded_extension": investigation(seed),
            "note": "Original uses data seed 20261004 and bootstrap seed 314159; --seed changes only the extension."}
