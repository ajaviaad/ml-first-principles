"""Readable NumPy experiments for Chapters 7–15.

Rows are observations; fit functions return ordinary dictionaries or arrays.
All losses explicitly state whether they sum or average observations. Small,
exhaustive implementations prioritize inspectability over production speed.
"""
from itertools import combinations
import numpy as np
from book_code import companion_classical as book


def _matrix(X, name="X"):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or min(X.shape) < 1 or not np.isfinite(X).all():
        raise ValueError(f"{name} must be a nonempty finite 2-D matrix")
    return X


def _xy(X, y, binary=False):
    X = _matrix(X)
    y = np.asarray(y, dtype=float)
    if y.shape != (len(X),) or not np.isfinite(y).all():
        raise ValueError("y must be a finite vector with one target per row")
    if binary and not np.isin(y, [0, 1]).all():
        raise ValueError("binary labels must be 0 or 1")
    return X, y


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative(value, name):
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def ridge_fit(X, y, penalty=0.0):
    """Return [intercept, slopes...] minimizing SSE + penalty*sum(slopes**2).

    Uses the book's augmented least-squares solver; the intercept is unpenalized.
    X has shape (n,p), y shape (n,); no automatic standardization is applied.
    """
    X, y = _xy(X, y)
    _nonnegative(penalty, "penalty")
    return book.ridge_fit(X, y, penalty)


def linear_predict(coefficients, X):
    """Apply coefficients returned by ridge_fit to an (n,p) matrix."""
    X = _matrix(X)
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.shape != (X.shape[1] + 1,) or not np.isfinite(coefficients).all():
        raise ValueError("coefficients must contain one intercept and p slopes")
    return coefficients[0] + X @ coefficients[1:]


def polynomial_features(x, degree=2):
    """Columns x, x**2, ..., x**degree; the fitting function adds an intercept."""
    _positive_int(degree, "degree")
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("x must be a nonempty finite vector")
    features = x[:, None] ** np.arange(1, degree + 1)
    if not np.isfinite(features).all():
        raise ValueError("polynomial powers overflow; rescale x or reduce degree")
    return features


def sigmoid(z):
    """Stable sigmoid; supports scalar or array inputs, including extreme logits."""
    return book.sigmoid(z)


def logistic_loss_gradient(design, y, coefficients, penalty=0.0):
    """Mean binary log loss + penalty/2 * ||slopes||², and its gradient.

    design includes an intercept in column zero; this coefficient is not penalized.
    The original book helper uses a SUM; division by n here is deliberate.
    """
    design, y = _xy(design, y, binary=True)
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.shape != (design.shape[1],) or not np.isfinite(coefficients).all():
        raise ValueError("coefficient shape must match design columns")
    if not np.all(design[:, 0] == 1):
        raise ValueError("design's first column must be ones for the intercept")
    _nonnegative(penalty, "penalty")
    loss, gradient = book.logistic_loss_gradient(design, y, coefficients)
    regularized = coefficients.copy()
    regularized[0] = 0
    return (float(loss / len(y) + penalty * regularized @ regularized / 2),
            gradient / len(y) + penalty * regularized)


def logistic_fit(X, y, penalty=0.01, learning_rate=0.2, steps=400):
    """Full-batch gradient descent; returns coefficients and objective history.

    No convergence guarantee for an arbitrary learning rate; inspect the history.
    Coefficients include an intercept. Both classes must occur in training data.
    """
    X, y = _xy(X, y, binary=True)
    if len(np.unique(y)) != 2:
        raise ValueError("both classes must be present")
    _positive_int(steps, "steps")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    design = np.column_stack([np.ones(len(X)), X])
    coefficients = np.zeros(design.shape[1])
    history = []
    for _ in range(steps):
        loss, gradient = logistic_loss_gradient(design, y, coefficients, penalty)
        history.append(loss)
        coefficients -= learning_rate * gradient
    history.append(logistic_loss_gradient(design, y, coefficients, penalty)[0])
    return {"coefficients": coefficients, "loss_history": np.asarray(history)}


def bernoulli_nb_fit(X, y, alpha=1.0):
    """Binary class Bernoulli naive Bayes with symmetric feature pseudocounts.

    Class priors are empirical; feature probabilities are (ones+alpha)/(n+2*alpha).
    Presence AND absence contribute to predictions. Both classes are required.
    """
    X, y = _xy(X, y, binary=True)
    if not np.isin(X, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("X must be binary and both classes must be present")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    counts = np.array([(y == c).sum() for c in (0, 1)])
    probabilities = np.array([(X[y == c].sum(axis=0) + alpha) /
                              (counts[c] + 2 * alpha) for c in (0, 1)])
    return {"class_prior": counts / len(y), "feature_probability": probabilities}


def bernoulli_nb_predict_proba(model, X):
    """Return (n,2) class probabilities by normalizing log-joint scores."""
    X = _matrix(X)
    probabilities = np.asarray(model["feature_probability"], dtype=float)
    priors = np.asarray(model["class_prior"], dtype=float)
    if not np.isin(X, [0, 1]).all() or probabilities.shape != (2, X.shape[1]):
        raise ValueError("X must be binary with the trained feature count")
    if priors.shape != (2,) or not np.isfinite(priors).all() or np.any(priors <= 0) or not np.isclose(priors.sum(), 1):
        raise ValueError("class priors must be positive and sum to one")
    if not np.isfinite(probabilities).all() or np.any((probabilities <= 0) | (probabilities >= 1)):
        raise ValueError("feature probabilities must be strictly between zero and one")
    log_joint = (X @ np.log(probabilities).T +
                 (1 - X) @ np.log1p(-probabilities).T + np.log(priors))
    mass = np.exp(log_joint - log_joint.max(axis=1, keepdims=True))
    return mass / mass.sum(axis=1, keepdims=True)


def knn_probability(X, y, query, k=3, weighted=False):
    """Binary local probability with stable input-order ties.

    If selected neighbors include exact matches, weighted mode averages only
    those matches. Duplicates beyond k are deliberately outside the neighborhood.
    """
    X, y = _xy(X, y, binary=True)
    _positive_int(k, "k")
    query = np.asarray(query, dtype=float)
    if query.shape != (X.shape[1],) or not np.isfinite(query).all():
        raise ValueError("query must have one finite value per feature")
    if k > len(X):
        raise ValueError("k cannot exceed the number of training observations")
    return book.neighbor_probability(X, y, query, k, weighted)


def hinge_loss_gradient(X, y, w, b=0.0, penalty=1.0):
    """Mean max(0,1-y*(Xw+b)) + penalty/2*||w||²; subgradient at hinge=0 is 0.

    y is encoded -1/+1. This exposes an SVM-style objective, not an SVM solver.
    """
    X, y = _xy(X, y)
    w = np.asarray(w, dtype=float)
    if not np.isin(y, [-1, 1]).all() or w.shape != (X.shape[1],):
        raise ValueError("y must be -1/+1 and w must match the features")
    if not np.isfinite(w).all() or not np.isfinite(b):
        raise ValueError("weights and bias must be finite")
    _nonnegative(penalty, "penalty")
    margin = y * (X @ w + b)
    active = margin < 1
    loss = np.maximum(0, 1 - margin).mean() + penalty * (w @ w) / 2
    gradient_w = penalty * w - X.T @ (y * active) / len(y)
    gradient_b = -float(np.mean(y * active))
    return float(loss), gradient_w, gradient_b


def rbf_kernel(X, Y, gamma=1.0):
    """Pairwise exp(-gamma*||x-y||²), shape (len(X),len(Y))."""
    X, Y = _matrix(X), _matrix(Y, "Y")
    _nonnegative(gamma, "gamma")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same feature count")
    squared = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2)
    return np.exp(-gamma * squared)


def _split_threshold(low, high):
    """Choose a finite threshold in [low,high) for distinct sorted finite values.

    Halving first avoids overflow when endpoints have opposite large signs.
    Adjacent floats may have no representable interior midpoint: falling back
    to low still separates them under the tree's ``value <= threshold`` test.
    """
    midpoint = low / 2 + high / 2
    return float(midpoint if low <= midpoint < high else low)


def _best_regression_split(X, y, min_leaf=1):
    """Shared exhaustive split search on already validated arrays."""
    best = None
    for feature in range(X.shape[1]):
        unique = np.unique(X[:, feature])
        for low, high in zip(unique[:-1], unique[1:]):
            threshold = _split_threshold(low, high)
            left = X[:, feature] <= threshold
            if min(left.sum(), (~left).sum()) < min_leaf:
                continue
            lmean, rmean = y[left].mean(), y[~left].mean()
            residual = y - np.where(left, lmean, rmean)
            error = float(residual @ residual)
            if best is None or error < best[0]:
                best = error, feature, threshold, float(lmean), float(rmean)
    return best


def regression_stump(X, y):
    """Best numeric split: (SSE, feature, threshold, left_mean, right_mean).

    This independent version of the book's exhaustive search uses thresholds
    safe for adjacent floating-point values and large finite feature values.
    Equal-quality splits follow feature order, then increasing threshold order.
    """
    X, y = _xy(X, y)
    best = _best_regression_split(X, y)
    if best is None:
        raise ValueError("No nonempty split is available")
    return best


def stump_predict(stump, X):
    """Predict with a stump returned by regression_stump."""
    X = _matrix(X)
    return book.stump_predict(stump, X)


def gini(labels):
    """Gini impurity 1-sum(class_fraction²), supporting arbitrary class labels."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or not len(labels):
        raise ValueError("labels must be a nonempty vector")
    _, count = np.unique(labels, return_counts=True)
    return float(1 - np.sum((count / len(labels)) ** 2))


def tree_fit(X, y, max_depth=2, min_leaf=1):
    """Greedy regression tree, exhaustive thresholds and strict positive gain.

    Depth zero is a constant model. A dictionary node is easy to inspect.
    Zero-gain root splits are refused, so a symmetric XOR problem is not solved.
    """
    X, y = _xy(X, y)
    if not isinstance(max_depth, (int, np.integer)) or max_depth < 0:
        raise ValueError("max_depth must be a nonnegative integer")
    _positive_int(min_leaf, "min_leaf")

    def grow(a, target, depth):
        node = {"value": float(target.mean()), "count": len(target),
                "n_features": X.shape[1]}
        baseline = float(np.sum((target - target.mean()) ** 2))
        if depth == 0 or len(a) < 2 * min_leaf:
            return node
        best = _best_regression_split(a, target, min_leaf)
        if best is None or best[0] >= baseline - 1e-12:
            return node
        error, feature, threshold, _, _ = best
        left = a[:, feature] <= threshold
        node.update(feature=feature, threshold=float(threshold), gain=baseline - error,
                    left=grow(a[left], target[left], depth - 1),
                    right=grow(a[~left], target[~left], depth - 1))
        return node

    return grow(X, y, max_depth)


def tree_predict(tree, X):
    """Evaluate each row by following learned numeric split conditions."""
    X = _matrix(X)
    if X.shape[1] != tree["n_features"]:
        raise ValueError("X must have the trained feature count")
    answer = []
    for row in X:
        node = tree
        while "feature" in node:
            node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
        answer.append(node["value"])
    return np.asarray(answer)


def bagging_fit(X, y, n_estimators=20, max_depth=2, seed=42):
    """Bootstrap regression trees; stores draw indices for out-of-bag inspection."""
    X, y = _xy(X, y)
    _positive_int(n_estimators, "n_estimators")
    rng = np.random.default_rng(seed)
    samples = [rng.integers(0, len(X), size=len(X)) for _ in range(n_estimators)]
    trees = [tree_fit(X[index], y[index], max_depth) for index in samples]
    return {"trees": trees, "bootstrap_indices": samples}


def bagging_predict(model, X):
    """Average all tree predictions, giving every fitted tree equal weight."""
    if not model["trees"]:
        raise ValueError("model must contain at least one tree")
    return np.mean([tree_predict(tree, X) for tree in model["trees"]], axis=0)


def boosting_fit(X, y, n_estimators=10, learning_rate=0.2):
    """Squared-loss gradient boosting with regression stumps and a mean baseline.

    A constant-feature dataset returns the baseline with no further stages.
    Restricting shrinkage to (0,1] makes exact residual fits nonincreasing in SSE.
    """
    X, y = _xy(X, y)
    _positive_int(n_estimators, "n_estimators")
    if not np.isfinite(learning_rate) or not 0 < learning_rate <= 1:
        raise ValueError("learning_rate must be in (0,1]")
    initial = float(y.mean())
    prediction = np.full(len(y), initial)
    stumps, errors = [], [float(np.sum((y - prediction) ** 2))]
    for _ in range(n_estimators):
        try:
            stump = regression_stump(X, y - prediction)
        except ValueError:  # no nonempty split; a constant fit is already optimal
            break
        prediction += learning_rate * stump_predict(stump, X)
        stumps.append(stump)
        errors.append(float(np.sum((y - prediction) ** 2)))
    return {"initial": initial, "learning_rate": learning_rate,
            "stumps": stumps, "sse_history": np.asarray(errors), "n_features": X.shape[1]}


def boosting_predict(model, X):
    """Add the baseline and every shrunken residual stump."""
    X = _matrix(X)
    if X.shape[1] != model["n_features"]:
        raise ValueError("X must have the trained feature count")
    prediction = np.full(len(X), model["initial"])
    for stump in model["stumps"]:
        prediction += model["learning_rate"] * stump_predict(stump, X)
    return prediction


def kmeans(X, centers, max_steps=100):
    """Lloyd's algorithm with explicit initial centers and objective trace.

    Empty clusters fail explicitly. Returns (labels, centers, SSE trace).
    Initialization is a modeling choice; no global-optimum claim is made.
    """
    X, centers = _matrix(X), _matrix(centers, "centers")
    _positive_int(max_steps, "max_steps")
    if X.shape[1] != centers.shape[1] or len(centers) > len(X):
        raise ValueError("centers must match features and cannot outnumber points")
    return book.kmeans(X, centers, max_steps)


def pca_fit(X, rank=2):
    """Centered PCA using SVD; returns a model reusable for unseen observations.

    Rows are samples; components has shape (rank,p). A constant matrix has
    explained_variance_ratio defined here as zero, since total variance is zero.
    """
    X = _matrix(X)
    _positive_int(rank, "rank")
    if len(X) < 2 or rank > min(X.shape):
        raise ValueError("PCA requires at least two rows and rank <= min(n,p)")
    _, _, singular, components = book.pca(X, rank)
    energy = singular ** 2
    ratio = energy / energy.sum() if energy.sum() > 0 else np.zeros_like(energy)
    return {"mean": X.mean(axis=0), "components": components,
            "singular_values": singular, "explained_variance": energy / (len(X) - 1),
            "explained_variance_ratio": ratio}


def pca_transform(model, X):
    """Project using the training mean and component directions."""
    X = _matrix(X)
    if X.shape[1] != len(model["mean"]):
        raise ValueError("X must have the trained feature count")
    return (X - model["mean"]) @ model["components"].T


def pca_inverse_transform(model, scores):
    """Reconstruct projected coordinates in the original feature units."""
    scores = _matrix(scores, "scores")
    if scores.shape[1] != len(model["components"]):
        raise ValueError("scores must have one column per retained component")
    return scores @ model["components"] + model["mean"]


def frequent_itemsets(transactions, minimum_count=2):
    """Apriori absolute support counts; transaction items must be sortable strings.

    A transaction is a set: repeated items count once. Empty transactions remain
    in the population denominator when computing rule probabilities.
    """
    _positive_int(minimum_count, "minimum_count")
    transactions = [frozenset(row) for row in transactions]
    if any(not isinstance(item, str) for row in transactions for item in row):
        raise ValueError("item labels must be strings")
    return book.frequent_itemsets(transactions, minimum_count)


def association_rules(transactions, minimum_count=2, min_confidence=0.5):
    """Generate rules with counts, support, confidence, lift and leverage.

    Only frequent unions are searched. Rule fields contain sorted item lists,
    giving deterministic ordering and JSON-serializable outputs.
    """
    if not np.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be between zero and one")
    transactions = [frozenset(row) for row in transactions]
    counts = frequent_itemsets(transactions, minimum_count)
    rules = []
    for union in sorted(counts, key=lambda value: (len(value), sorted(value))):
        for size in range(1, len(union)):
            for items in combinations(sorted(union), size):
                antecedent = frozenset(items)
                consequent = union - antecedent
                confidence = counts[union] / counts[antecedent]
                if confidence < min_confidence:
                    continue
                support = counts[union] / len(transactions)
                p_a, p_b = counts[antecedent] / len(transactions), counts[consequent] / len(transactions)
                rules.append({"antecedent": sorted(antecedent), "consequent": sorted(consequent),
                              "count": counts[union], "support": support, "confidence": confidence,
                              "lift": confidence / p_b, "leverage": support - p_a * p_b})
    return rules


def _hmm_inputs(initial, transition, emission, observations):
    initial = np.asarray(initial, dtype=float)
    transition, emission = _matrix(transition, "transition"), _matrix(emission, "emission")
    if initial.ndim != 1 or len(initial) < 1 or not np.isfinite(initial).all():
        raise ValueError("initial must be a finite probability vector")
    k = len(initial)
    if transition.shape != (k, k) or emission.shape[0] != k:
        raise ValueError("HMM state dimensions do not match")
    if np.any(initial < 0) or not np.isclose(initial.sum(), 1):
        raise ValueError("initial probabilities must be nonnegative and sum to one")
    for matrix in (transition, emission):
        if np.any(matrix < 0) or not np.allclose(matrix.sum(axis=1), 1):
            raise ValueError("transition and emission rows must be probability vectors")
    observations = list(observations)
    if any(isinstance(o, bool) or not isinstance(o, (int, np.integer)) or
           not 0 <= o < emission.shape[1] for o in observations):
        raise ValueError("observations must be valid integer emission indices")
    return initial, transition, emission, observations


def hmm_forward(initial, transition, emission, observations):
    """Scaled categorical filtering; returns (T,K) probabilities and log evidence.

    transition[i,j]=P(next=j|current=i); emission[j,o]=P(observation=o|state=j).
    An empty sequence has log evidence 0 and an empty posterior matrix.
    """
    inputs = _hmm_inputs(initial, transition, emission, observations)
    return book.hmm_forward(*inputs)


def hmm_smooth(initial, transition, emission, observations):
    """Forward-backward smoothed state marginals using all observations."""
    initial, transition, emission, observations = _hmm_inputs(initial, transition, emission, observations)
    filtered, _ = book.hmm_forward(initial, transition, emission, observations)
    smoothed = filtered.copy()
    backward = np.ones(len(initial))
    for t in range(len(observations) - 2, -1, -1):
        backward = transition @ (emission[:, observations[t + 1]] * backward)
        backward /= backward.max()  # any positive scaling cancels when normalizing
        mass = filtered[t] * backward
        smoothed[t] = mass / mass.sum()
    return smoothed


def hmm_viterbi(initial, transition, emission, observations):
    """Best joint state path and its joint log probability, using max-product.

    Zero probabilities become -infinity. Ties choose the lowest state index.
    The result is not a sequence of independent marginal winners.
    """
    initial, transition, emission, observations = _hmm_inputs(initial, transition, emission, observations)
    if not observations:
        return [], 0.0
    with np.errstate(divide="ignore"):
        log_initial, log_transition, log_emission = np.log(initial), np.log(transition), np.log(emission)
    score = log_initial + log_emission[:, observations[0]]
    predecessors = []
    for observation in observations[1:]:
        candidates = score[:, None] + log_transition
        previous = candidates.argmax(axis=0)
        predecessors.append(previous)
        score = candidates[previous, np.arange(len(initial))] + log_emission[:, observation]
    state = int(score.argmax())
    if not np.isfinite(score[state]):
        raise ValueError("Observation sequence has zero probability under the model")
    best_score = float(score[state])
    path = [state]
    for previous in reversed(predecessors):
        state = int(previous[state])
        path.append(state)
    return path[::-1], best_score


def lesson_07(seed=42):
    """Closed-form line, ridge shrinkage and a quadratic holdout comparison."""
    X, y = np.array([[0.], [1.], [2.]]), np.array([1., 2., 2.])
    coefficients = ridge_fit(X, y)
    residual = y - linear_predict(coefficients, X)
    rng = np.random.default_rng(seed)
    train = np.linspace(-2, 2, 30)
    target = 1 + 0.5 * train + train ** 2 + rng.normal(0, 0.15, len(train))
    test = np.linspace(-1.9, 1.9, 40)
    truth = 1 + 0.5 * test + test ** 2
    line = ridge_fit(train[:, None], target)
    quadratic = ridge_fit(polynomial_features(train), target)
    return {"coefficients": coefficients.tolist(), "residuals": residual.tolist(),
            "sse": float(residual @ residual), "residual_sum": float(residual.sum()),
            "ridge_coefficients_lambda_2": ridge_fit(X, y, 2).tolist(),
            "heldout_line_mse": float(np.mean((linear_predict(line, test[:, None]) - truth) ** 2)),
            "heldout_quadratic_mse": float(np.mean((linear_predict(quadratic, polynomial_features(test)) - truth) ** 2))}


def lesson_08(seed=42):
    """Regularized logistic training and the archive naive-Bayes calculation."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(160, 2))
    y = (rng.random(160) < sigmoid(1.8 * X[:, 0] - X[:, 1])).astype(float)
    model = logistic_fit(X[:120], y[:120])
    probability = sigmoid(linear_predict(model["coefficients"], X[120:]))
    archive = {"class_prior": np.array([0.75, 0.25]),
               "feature_probability": np.array([[0.1, 0.2], [0.8, 0.6]])}
    posterior = bernoulli_nb_predict_proba(archive, [[1, 1]])
    return {"initial_loss": float(model["loss_history"][0]),
            "final_loss": float(model["loss_history"][-1]),
            "coefficients": model["coefficients"].tolist(),
            "test_accuracy": float(np.mean((probability >= 0.5) == y[120:])),
            "test_brier_score": float(np.mean((probability - y[120:]) ** 2)),
            "archive_urgent_probability": float(posterior[0, 1]),
            "archive_duplicate_deadline_probability": 64 / 65,
            "sigmoid_extremes": sigmoid([-1000, 0, 1000]).tolist()}


def lesson_09(seed=42):
    """Neighborhood sensitivity, a margin calculation and a kernel Gram matrix."""
    X, y = np.array([[0.], [1.], [2.], [6.], [8.]]), np.array([0, 0, 0, 1, 1])
    gram = rbf_kernel(X, X, gamma=0.5)
    loss, gradient, bias_gradient = hinge_loss_gradient([[0.], [2.]], [-1, 1], [1.], b=-1, penalty=1)
    return {"query": 4.4, "probability_k1": knn_probability(X, y, [4.4], 1),
            "probability_k3": knn_probability(X, y, [4.4], 3),
            "probability_weighted_k3": knn_probability(X, y, [4.4], 3, True),
            "neighborhood_radius_k3": float(np.sort(np.abs(X[:, 0] - 4.4))[2]),
            "margin_width": 2.0, "regularized_hinge_objective": loss,
            "hinge_weight_subgradient": gradient.tolist(), "hinge_bias_subgradient": bias_gradient,
            "rbf_smallest_eigenvalue": float(np.linalg.eigvalsh(gram).min())}


def lesson_10(seed=42):
    """Exact stump, a recursive tree, and the greedy XOR limitation."""
    X, y = np.arange(1, 5, dtype=float)[:, None], np.array([1., 2., 8., 9.])
    stump = regression_stump(X, y)
    tree = tree_fit(X, y, max_depth=2)
    xor_X, xor_y = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 1, 1, 0])
    xor_tree = tree_fit(xor_X, xor_y, 2)
    return {"baseline_sse": float(np.sum((y - y.mean()) ** 2)),
            "stump": list(stump), "stump_predictions": stump_predict(stump, X).tolist(),
            "depth2_predictions": tree_predict(tree, X).tolist(),
            "root_gini_6_vs_4": gini([1] * 6 + [0] * 4),
            "xor_greedy_predictions": tree_predict(xor_tree, xor_X).tolist(), "tree": tree}


def lesson_11(seed=42):
    """Residual boosting arithmetic and bootstrap diversity."""
    X, y = np.arange(1, 5, dtype=float)[:, None], np.array([2., 2., 6., 6.])
    boosted = boosting_fit(X, y, n_estimators=2, learning_rate=0.5)
    bagged = bagging_fit(X, y, n_estimators=20, max_depth=1, seed=seed)
    return {"boosting_sse_history": boosted["sse_history"].tolist(),
            "boosting_predictions": boosting_predict(boosted, X).tolist(),
            "bagging_predictions": bagging_predict(bagged, X).tolist(),
            "first_bootstrap_indices": bagged["bootstrap_indices"][0].tolist(),
            "ensemble_size": len(bagged["trees"]),
            "mean_bootstrap_unique_fraction": float(np.mean([len(np.unique(i)) / len(i) for i in bagged["bootstrap_indices"]])),
            "variance_formula_rho_0_2_B_20": 0.2 + 0.8 / 20}


def lesson_12(seed=42):
    """Lloyd iteration with two initializations revealing a local optimum."""
    X = np.array([[0.], [1.], [4.], [9.]])
    labels, centers, trace = kmeans(X, [[0.], [9.]])
    alternate_labels, alternate_centers, alternate_trace = kmeans(X, [[0.], [4.]])
    return {"labels": labels.tolist(), "centers": centers[:, 0].tolist(),
            "objective_trace": trace, "alternate_labels": alternate_labels.tolist(),
            "alternate_centers": alternate_centers[:, 0].tolist(),
            "alternate_objective_trace": alternate_trace,
            "preferred_run": "first" if trace[-1] < alternate_trace[-1] else "alternate"}


def lesson_13(seed=42):
    """PCA reconstruction equals discarded singular energy."""
    X = np.array([[2., 0., 1.], [-2., 0., -1.], [0., 1., 0.], [0., -1., 0.]])
    model = pca_fit(X, rank=1)
    scores = pca_transform(model, X)
    reconstructed = pca_inverse_transform(model, scores)
    return {"singular_values": model["singular_values"].tolist(),
            "first_component": model["components"][0].tolist(), "scores": scores[:, 0].tolist(),
            "retained_variance_fraction": float(model["explained_variance_ratio"][0]),
            "reconstruction_sse": float(np.sum((X - reconstructed) ** 2)),
            "discarded_singular_energy": float(np.sum(model["singular_values"][1:] ** 2)),
            "covariance_example_retained_fraction": 7 / 8}


def lesson_14(seed=42):
    """Mine baskets and inspect baseline-adjusted association strength."""
    baskets = [{"tea", "sugar", "biscuits"}, {"tea", "biscuits"}, {"tea", "sugar"},
               {"coffee", "sugar"}, {"tea", "biscuits"}, {"coffee", "biscuits"}]
    counts = frequent_itemsets(baskets, 2)
    rules = association_rules(baskets, 2, 0.5)
    selected = next(rule for rule in rules if rule["antecedent"] == ["tea"] and rule["consequent"] == ["biscuits"])
    return {"transaction_count": len(baskets),
            "frequent_itemsets": [{"items": sorted(items), "count": count}
                                  for items, count in sorted(counts.items(), key=lambda pair: (len(pair[0]), sorted(pair[0])))],
            "tea_to_biscuits": selected, "rules": rules}


def lesson_15(seed=42):
    """Filtering, smoothing and whole-path decoding for two machine states."""
    initial, transition, emission = [.9, .1], [[.95, .05], [.1, .9]], [[.9, .1], [.2, .8]]
    observations = [1, 1]
    filtered, log_likelihood = hmm_forward(initial, transition, emission, observations)
    smoothed = hmm_smooth(initial, transition, emission, observations)
    path, log_joint = hmm_viterbi(initial, transition, emission, observations)
    return {"observations": observations, "state_names": ["normal", "degraded"],
            "filtered_probabilities": filtered.tolist(), "smoothed_probabilities": smoothed.tolist(),
            "sequence_probability": float(np.exp(log_likelihood)), "log_likelihood": log_likelihood,
            "viterbi_path": path, "viterbi_state_names": [["normal", "degraded"][i] for i in path],
            "viterbi_joint_probability": float(np.exp(log_joint))}
