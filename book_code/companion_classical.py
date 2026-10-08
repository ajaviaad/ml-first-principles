# companion_classical.py
"""Companion experiments for Chapters 7-15.

Requires Python 3.11+ and NumPy. Run: python companion_classical.py
All data are constructed. No downloads or external files are used.
These transparent implementations teach mechanisms; they are not optimized
replacements for maintained libraries. Tests compare results with closed-form
answers, finite differences, or exhaustive enumeration on small cases.
"""
from itertools import combinations, product
import numpy as np


def ridge_fit(X, y, penalty=0.0):
    """Squared-error SUM plus penalty * ||coefficients||^2; free intercept."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or y.shape != (len(X),) or penalty < 0:
        raise ValueError(
            "Expected a matrix, matching response vector, "
            "and nonnegative penalty"
        )
    design = np.column_stack([np.ones(len(X)), X])
    reg = np.eye(design.shape[1]) * np.sqrt(penalty)
    reg[0, 0] = 0.0
    return np.linalg.lstsq(
        np.vstack([design, reg]),
        np.r_[y, np.zeros(design.shape[1])],
        rcond=None,
    )[0]


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    answer = np.empty_like(z)
    positive = z >= 0
    answer[positive] = 1 / (1 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    answer[~positive] = exp_z / (1 + exp_z)
    return answer


def logistic_loss_gradient(X, y, w):
    """Sum binary log loss and its gradient, without regularization."""
    z = X @ w
    loss = np.sum(np.logaddexp(0.0, z) - y * z)
    gradient = X.T @ (sigmoid(z) - y)
    return float(loss), gradient


def neighbor_probability(X, y, query, k, weighted=False):
    """Binary local class fraction. Equal-distance ties follow input order."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    if not 1 <= k <= len(X):
        raise ValueError(
            "k must be between one and the number of observations"
        )
    distances = np.linalg.norm(X - np.asarray(query, float), axis=1)
    nearest = np.argsort(distances, kind="stable")[:k]
    d, labels = distances[nearest], y[nearest]
    if weighted:
        if np.any(d == 0):
            return float(labels[d == 0].mean())
        return float(np.average(labels, weights=1 / d))
    return float(labels.mean())


def regression_stump(X, y):
    """Find the best nonempty numeric split through exhaustive search."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    best = None
    for feature in range(X.shape[1]):
        values = np.unique(X[:, feature])
        for low, high in zip(values[:-1], values[1:]):
            threshold = low + (high - low) / 2
            left = X[:, feature] <= threshold
            lmean, rmean = y[left].mean(), y[~left].mean()
            prediction = np.where(left, lmean, rmean)
            error = float(np.sum((y - prediction) ** 2))
            if best is None or error < best[0]:
                best = error, feature, threshold, lmean, rmean
    if best is None:
        raise ValueError("No nonempty split is available")
    return best


def stump_predict(stump, X):
    _, feature, threshold, lmean, rmean = stump
    return np.where(np.asarray(X)[:, feature] <= threshold, lmean, rmean)


def kmeans(X, centers, max_steps=100):
    """Lloyd updates; fail explicitly on empty groups."""
    X, centers = np.asarray(X, float), np.array(centers, float, copy=True)
    trace = []
    for _ in range(max_steps):
        squared = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = squared.argmin(axis=1)
        trace.append(float(squared[np.arange(len(X)), labels].sum()))
        if any(not np.any(labels == k) for k in range(len(centers))):
            raise ValueError("Empty cluster; choose another initialization")
        updated = np.array([
            X[labels == k].mean(axis=0) for k in range(len(centers))
        ])
        if np.array_equal(updated, centers):
            return labels, centers, trace
        centers = updated
    raise RuntimeError("Iteration limit reached")


def pca(X, rank):
    X = np.asarray(X, float)
    centered = X - X.mean(axis=0)
    U, singular, Vt = np.linalg.svd(centered, full_matrices=False)
    if not 1 <= rank <= len(singular):
        raise ValueError("rank is outside the available matrix dimensions")
    scores = centered @ Vt[:rank].T
    reconstruction = scores @ Vt[:rank] + X.mean(axis=0)
    return scores, reconstruction, singular, Vt[:rank]


def frequent_itemsets(transactions, minimum_count):
    """Apriori with absolute support counts; item labels must be sortable."""
    if minimum_count < 1:
        raise ValueError("minimum_count must be positive")
    transactions = [frozenset(t) for t in transactions]
    universe = sorted(set().union(*transactions)) if transactions else []
    candidates = {frozenset([item]) for item in universe}
    result = {}
    size = 1
    while candidates:
        counts = {c: sum(c <= t for t in transactions) for c in candidates}
        level = {c for c, count in counts.items() if count >= minimum_count}
        result.update({c: counts[c] for c in level})
        size += 1
        joined = {a | b for a in level for b in level if len(a | b) == size}
        candidates = {
            c for c in joined
            if all(
                frozenset(s) in level
                for s in combinations(sorted(c), size - 1)
            )
        }
    return result


def hmm_forward(initial, transition, emission, observations):
    """Categorical forward filtering with scaling and log likelihood."""
    initial, transition, emission = map(
        lambda a: np.asarray(a, float),
        (initial, transition, emission),
    )
    if not observations:
        return np.empty((0, len(initial))), 0.0
    posterior, log_likelihood, filtered = initial.copy(), 0.0, []
    for t, observation in enumerate(observations):
        prior = posterior if t == 0 else posterior @ transition
        mass = prior * emission[:, observation]
        scale = mass.sum()
        if scale <= 0:
            raise ValueError(
                "Observation sequence has zero probability under the model"
            )
        posterior = mass / scale
        log_likelihood += np.log(scale)
        filtered.append(posterior.copy())
    return np.array(filtered), float(log_likelihood)


def hmm_enumerate(initial, transition, emission, observations):
    """Independent, exponential-time reference calculation for tiny tests."""
    probabilities = []
    paths = list(product(range(len(initial)), repeat=len(observations)))
    for path in paths:
        probability = initial[path[0]] * emission[path[0], observations[0]]
        for t in range(1, len(path)):
            probability *= (
                transition[path[t - 1], path[t]]
                * emission[path[t], observations[t]]
            )
        probabilities.append(probability)
    return paths, np.asarray(probabilities)


def run_experiments():
    np.set_printoptions(precision=6, suppress=True)
    X = np.array([[0.], [1.], [2.]])
    y = np.array([1., 2., 2.])
    coefficients = ridge_fit(X, y)
    np.testing.assert_allclose(coefficients, [7 / 6, 1 / 2])
    residual = y - np.column_stack([np.ones(3), X]) @ coefficients
    np.testing.assert_allclose(np.sum(residual ** 2), 1 / 6)
    np.testing.assert_allclose(
        ridge_fit(np.ones((3, 1)), np.full(3, 7.), 100),
        [7, 0],
        atol=1e-12,
    )
    print(
        "Chapter 7: intercept and slope",
        coefficients, "SSE", residual @ residual,
    )

    design = np.array([[1., -2.], [1., .5], [1., 3.]])
    binary = np.array([0., 1., 1.])
    weights = np.array([.3, -.2])
    _, analytic = logistic_loss_gradient(design, binary, weights)
    numerical = []
    for j in range(len(weights)):
        step = np.zeros_like(weights)
        step[j] = 1e-6
        high, _ = logistic_loss_gradient(design, binary, weights + step)
        low, _ = logistic_loss_gradient(design, binary, weights - step)
        numerical.append((high - low) / 2e-6)
    np.testing.assert_allclose(analytic, numerical, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(
        sigmoid(np.array([-1000., 0., 1000.])), [0., .5, 1.]
    )
    print("Chapter 8: logistic gradient verified against finite differences")

    points = np.array([[1.], [2.], [3.]])
    labels = np.array([1., 0., 0.])
    plain = neighbor_probability(points, labels, [0.], 3)
    weighted = neighbor_probability(points, labels, [0.], 3, weighted=True)
    np.testing.assert_allclose(plain, 1 / 3)
    np.testing.assert_allclose(weighted, 6 / 11)
    print(
        "Chapter 9: class B fraction, equal and distance weighted",
        plain, weighted,
    )

    fruit = np.arange(1, 5, dtype=float)[:, None]
    sweetness = np.array([1., 2., 8., 9.])
    stump = regression_stump(fruit, sweetness)
    np.testing.assert_allclose(stump, [1., 0, 2.5, 1.5, 8.5])
    print(
        "Chapter 10: best stump (SSE, feature, threshold, left, right)",
        stump,
    )
    targets = np.array([2., 2., 6., 6.])
    prediction = np.full(4, targets.mean())
    errors = [float(np.sum((targets - prediction) ** 2))]
    for _ in range(2):
        component = regression_stump(fruit, targets - prediction)
        prediction += .5 * stump_predict(component, fruit)
        errors.append(float(np.sum((targets - prediction) ** 2)))
    np.testing.assert_allclose(errors, [16, 4, 1])
    print("Chapter 11: squared errors through two boosting stages", errors)

    values = np.array([[0.], [1.], [4.], [9.]])
    assignment, centers, trace = kmeans(values, [[0.], [9.]])
    np.testing.assert_array_equal(assignment, [0, 0, 0, 1])
    np.testing.assert_allclose(centers[:, 0], [5 / 3, 9])
    np.testing.assert_allclose(trace[-1], 26 / 3)
    assert all(a >= b - 1e-12 for a, b in zip(trace[:-1], trace[1:]))
    print(
        "Chapter 12: k-means centers and final objective",
        centers[:, 0], trace[-1],
    )

    covariance = np.array([[4., 3.], [3., 4.]])
    eigenvalues, _ = np.linalg.eigh(covariance)
    np.testing.assert_allclose(eigenvalues, [1., 7.])
    example = np.array([
        [2., 0., 1.], [-2., 0., -1.], [0., 1., 0.], [0., -1., 0.]
    ])
    _, reconstructed, singular, _ = pca(example, 1)
    np.testing.assert_allclose(
        np.sum((example - reconstructed) ** 2),
        np.sum(singular[1:] ** 2),
    )
    print(
        "Chapter 13: PCA reconstruction error equals "
        "discarded singular energy"
    )

    baskets = [set(['tea', 'sugar', 'biscuits']), set(['tea', 'biscuits']),
               set(['tea', 'sugar']), set(['coffee', 'sugar']),
               set(['tea', 'biscuits']), set(['coffee', 'biscuits'])]
    mined = frequent_itemsets(baskets, 2)
    items = sorted(set().union(*baskets))
    exhaustive = {}
    for size in range(1, len(items) + 1):
        for itemset in combinations(items, size):
            key = frozenset(itemset)
            count = sum(key <= basket for basket in baskets)
            if count >= 2:
                exhaustive[key] = count
    assert mined == exhaustive
    assert mined[frozenset(['tea', 'biscuits'])] == 3
    print(
        "Chapter 14: Apriori matches exhaustive enumeration; "
        "frequent itemsets",
        len(mined),
    )

    initial = np.array([.9, .1])
    transition = np.array([[.95, .05], [.1, .9]])
    emission = np.array([[.9, .1], [.2, .8]])
    observed = [1, 1]
    filtered, log_likelihood = hmm_forward(
        initial, transition, emission, observed
    )
    np.testing.assert_allclose(filtered[0, 1], 8 / 17)
    np.testing.assert_allclose(filtered[1, 1], .36 / .415)
    paths, probability = hmm_enumerate(
        initial, transition, emission, observed
    )
    np.testing.assert_allclose(np.exp(log_likelihood), probability.sum())
    joint_final = sum(
        p for path, p in zip(paths, probability) if path[-1] == 1
    )
    np.testing.assert_allclose(
        filtered[-1, 1], joint_final / probability.sum()
    )
    print(
        "Chapter 15: degraded-state filtering probabilities", filtered[:, 1]
    )
    print("All companion checks passed.")


if __name__ == '__main__':
    run_experiments()
# End of companion_classical.py
