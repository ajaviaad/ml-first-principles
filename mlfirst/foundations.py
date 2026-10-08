"""Chapters 1–6: data boundaries, calculus, probability and measurement.

Arrays use one row per observation; vector targets always have shape (n,).
The functions favor explicit assumptions and readable arithmetic over speed.
"""

from dataclasses import dataclass
import numpy as np


def vector(x, name="values"):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError(f"{name} must be a finite, nonempty vector")
    return x


def regression_metrics(y, prediction):
    """MAE and RMSE in target units; R² is None for a constant target."""
    y, prediction = vector(y, "y"), vector(prediction, "prediction")
    if y.shape != prediction.shape:
        raise ValueError("targets and predictions must have the same shape")
    error = prediction - y
    denominator = float(np.sum((y - y.mean()) ** 2))
    return {"mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "r2": 1 - float(error @ error) / denominator if denominator else None}


@dataclass
class Standardizer:
    """Fit training statistics once. Constant columns receive scale 1."""
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or 0 in x.shape or not np.isfinite(x).all():
            raise ValueError("x must be a finite nonempty matrix")
        self.mean_ = x.mean(axis=0)
        scale = x.std(axis=0)
        self.scale_ = np.where(scale > 0, scale, 1.0)
        return self

    def transform(self, x):
        if self.mean_ is None:
            raise ValueError("fit on training data before transforming")
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.mean_) or not np.isfinite(x).all():
            raise ValueError("x must be finite with the fitted number of columns")
        return (x - self.mean_) / self.scale_


def split_groups(groups, test_fraction=0.25, seed=42):
    """Split whole groups; fraction applies to unique groups, not rows.

    Missing IDs are rejected: NaN does not compare equal to itself and can
    otherwise silently yield an empty holdout when used as a group label.
    """
    groups = np.asarray(groups)
    if groups.ndim != 1 or not 0 < test_fraction < 1:
        raise ValueError("expected group vector and fraction between zero and one")
    if groups.dtype.kind in "biufc" and not np.isfinite(groups).all():
        raise ValueError("group IDs must not be missing or nonfinite")
    if groups.dtype.kind in "Mm" and np.isnat(groups).any():
        raise ValueError("group IDs must not contain missing dates")
    if groups.dtype.kind == "O" and any(
        value is None or (isinstance(value, (float, complex, np.number)) and not np.isfinite(value))
        or (isinstance(value, (np.datetime64, np.timedelta64)) and np.isnat(value))
        for value in groups
    ):
        raise ValueError("group IDs must not be missing or nonfinite")
    try:
        unique = np.unique(groups)
    except TypeError as error:
        raise ValueError("group IDs must have mutually comparable scalar types") from error
    if len(unique) < 2:
        raise ValueError("at least two groups are required")
    shuffled = np.random.default_rng(seed).permutation(unique)
    n_test = min(len(unique) - 1, max(1, int(np.ceil(len(unique) * test_fraction))))
    mask = np.isin(groups, shuffled[:n_test])
    return np.flatnonzero(~mask), np.flatnonzero(mask)


def central_difference(function, parameters, epsilon=1e-5):
    """Numerical gradient of a scalar function; not a training algorithm."""
    parameters = vector(parameters, "parameters")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    gradient = np.empty_like(parameters)
    for j in range(len(parameters)):
        step = np.zeros_like(parameters)
        step[j] = epsilon
        gradient[j] = (function(parameters + step) - function(parameters - step)) / (2 * epsilon)
    return gradient


def logsumexp(values):
    values = vector(values)
    maximum = values.max()
    return float(maximum + np.log(np.exp(values - maximum).sum()))


def bayes_positive(prevalence, sensitivity, false_positive_rate):
    """P(defect | flag) from P(defect), P(flag | defect), P(flag | sound)."""
    if not all(np.isfinite(p) and 0 <= p <= 1 for p in (prevalence, sensitivity, false_positive_rate)):
        raise ValueError("all probabilities must be finite and in [0, 1]")
    true_flags = prevalence * sensitivity
    evidence = true_flags + (1 - prevalence) * false_positive_rate
    if evidence == 0:
        raise ValueError("conditioning event has zero probability")
    return true_flags / evidence


def bootstrap_mean(values, seed=42, repetitions=2000, level=0.95):
    """Percentile interval for an iid mean; unsuitable for dependent rows."""
    values = vector(values)
    if not isinstance(repetitions, int) or repetitions < 2 or not 0 < level < 1:
        raise ValueError("need at least two repetitions and a level in (0, 1)")
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
    return np.quantile(means, [(1 - level) / 2, (1 + level) / 2]).tolist()


def quadratic_descent(curvature=2.0, rate=0.2, initial=3.0, steps=12):
    """Trajectory for J(w)=curvature*w²/2, including initial w."""
    if not np.isfinite([curvature, rate, initial]).all() or curvature <= 0 or rate <= 0:
        raise ValueError("positive finite curvature and rate are required")
    if not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    path = [float(initial)]
    for _ in range(steps):
        path.append(path[-1] - rate * curvature * path[-1])
    return np.array(path)


def binary_metrics(y, probability, threshold=0.5):
    """Undefined precision/recall are None. ROC AUC includes half-credit ties.

    AUC uses pairwise comparison for clarity and quadratic memory; tiny data only.
    Probabilities are clipped only for log loss, not for decisions or Brier score.
    """
    y, p = vector(y, "y"), vector(probability, "probability")
    if y.shape != p.shape or not np.isin(y, [0, 1]).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError("matching binary labels and probabilities in [0, 1] required")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    positive = p >= threshold
    tp = int(np.sum(positive & (y == 1)))
    fp = int(np.sum(positive & (y == 0)))
    fn = int(np.sum(~positive & (y == 1)))
    tn = int(np.sum(~positive & (y == 0)))
    a, b = p[y == 1], p[y == 0]
    auc = float(np.mean((a[:, None] > b) + 0.5 * (a[:, None] == b))) if len(a) and len(b) else None
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": (tp + tn) / len(y),
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "brier": float(np.mean((p - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))),
            "roc_auc": auc}


def conformal_radius(residuals, alpha=0.1):
    """Split-conformal absolute-residual quantile; +inf for too little data."""
    residuals = vector(residuals, "residuals")
    if np.any(residuals < 0) or not 0 < alpha < 1:
        raise ValueError("nonnegative residuals and alpha in (0, 1) required")
    rank = int(np.ceil((len(residuals) + 1) * (1 - alpha)))
    return float(np.sort(residuals)[rank - 1]) if rank <= len(residuals) else float("inf")


def lesson_01(seed=42):
    rng = np.random.default_rng(seed)
    weekend = np.tile([0, 0, 0, 0, 0, 1, 1], 60)
    demand = 80 + 30 * weekend + rng.normal(0, 6, len(weekend))
    boundary = 280
    training_mean = demand[:boundary].mean()
    group_means = [demand[:boundary][weekend[:boundary] == k].mean() for k in (0, 1)]
    baseline = np.full(len(demand) - boundary, training_mean)
    prediction = np.array(group_means)[weekend[boundary:]]
    censored = np.minimum(demand, 90)
    return {"train_rows": boundary, "test_rows": len(prediction),
            "constant_baseline": regression_metrics(demand[boundary:], baseline),
            "weekday_model": regression_metrics(demand[boundary:], prediction),
            "training_means_by_weekend": group_means,
            "mean_demand": float(demand.mean()), "mean_observed_sales_if_stock_90": float(censored.mean())}


def lesson_02(seed=42):
    groups = np.repeat(np.arange(12), 4)
    rng = np.random.default_rng(seed)
    x = np.column_stack([groups + rng.normal(0, 0.1, len(groups)), np.ones(len(groups))])
    train, test = split_groups(groups, seed=seed)
    scaler = Standardizer().fit(x[train])
    before = scaler.mean_.copy()
    scaled_test = scaler.transform(x[test])
    return {"train_indices": train.tolist(), "test_indices": test.tolist(),
            "shared_groups": np.intersect1d(groups[train], groups[test]).tolist(),
            "fitted_mean": before.tolist(), "fitted_scale": scaler.scale_.tolist(),
            "train_scaled_mean": scaler.transform(x[train]).mean(axis=0).tolist(),
            "test_scaled_mean": scaled_test.mean(axis=0).tolist(),
            "transform_changed_fit": bool(np.any(scaler.mean_ != before)),
            "chronological_example": {"train": list(range(8)), "gap": [8, 9], "test": [10, 11]}}


def lesson_03(seed=42):
    x = np.array([[2., 3.], [0., 1.], [4., 2.]])
    w = np.array([4., -1.])
    objective = lambda theta: 0.5 * (2 * theta[0] + theta[1] - 5) ** 2
    theta = np.array([1., 0.])
    analytic = np.array([-6., -3.])
    numerical = central_difference(objective, theta)
    return {"X_shape": list(x.shape), "prediction": (x @ w + 7).tolist(),
            "analytic_gradient": analytic.tolist(), "numeric_gradient": numerical.tolist(),
            "gradient_max_error": float(np.max(np.abs(analytic - numerical))),
            "loss_before": objective(theta), "loss_after": objective(theta - 0.1 * analytic),
            "stable_logsumexp_1000_999": logsumexp([1000, 999])}


def lesson_04(seed=42):
    observations = np.array([1] * 8 + [0] * 2, dtype=float)
    p = np.array([0.5, 0.5])
    q = np.array([0.9, 0.1])
    entropy = float(-np.sum(p * np.log(p)))
    divergence = float(np.sum(p * np.log(p / q)))
    return {"defect_given_flag": bayes_positive(.01, .9, .05),
            "bernoulli_mle": float(observations.mean()),
            "beta_posterior_parameters": [9, 3], "posterior_mean": 9 / 12,
            "bootstrap_mean_interval": bootstrap_mean(observations, seed=seed),
            "entropy_nats": entropy, "kl_nats": divergence,
            "cross_entropy_nats": float(-np.sum(p * np.log(q)))}


def lesson_05(seed=42):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1, 1, 18)
    y = 1 + 2*x + rng.normal(0, .35, len(x))
    vx = np.linspace(-1, 1, 101)
    vy = 1 + 2*vx + rng.normal(0, .35, len(vx))
    scores = {}
    for degree in (1, 5, 15):
        train_design = np.polynomial.polynomial.polyvander(x, degree)
        validation_design = np.polynomial.polynomial.polyvander(vx, degree)
        coefficients = np.linalg.lstsq(train_design, y, rcond=None)[0]
        scores[str(degree)] = {"train_rmse": regression_metrics(y, train_design @ coefficients)["rmse"],
                              "validation_rmse": regression_metrics(vy, validation_design @ coefficients)["rmse"]}
    return {"stable_path": quadratic_descent(rate=.2).tolist(),
            "unstable_path": quadratic_descent(rate=1.1).tolist(),
            "degree_scores": scores,
            "selected_degree": int(min(scores, key=lambda d: scores[d]["validation_rmse"]))}


def lesson_06(seed=42):
    y = np.array([1] * 40 + [1] * 10 + [0] * 40 + [0] * 910)
    p = np.array([.8] * 40 + [.2] * 10 + [.8] * 40 + [.05] * 910)
    cost_threshold = 1 / (1 + 9)
    return {"threshold_0_5": binary_metrics(y, p),
            "threshold_for_fp_cost_1_fn_cost_9": cost_threshold,
            "cost_sensitive": binary_metrics(y, p, cost_threshold),
            "conformal_rank_for_19_at_90_percent": 18,
            "conformal_radius": conformal_radius(np.arange(1, 20) / 5),
            "conformal_width": 2 * conformal_radius(np.arange(1, 20) / 5)}
