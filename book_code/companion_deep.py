# companion_deep.py
"""Numerical companion to Chapters 16-24.

Requires Python 3.11+ and NumPy. Run: python companion_deep.py
No downloads, accelerators, or external datasets are needed.

The XOR fit illustrates optimization on four explicitly constructed points.
It is not a test of generalization. Other checks verify the book's arithmetic,
analytic derivatives, attention masks, and graph permutation behavior.
"""
from __future__ import annotations

import math
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Stable logistic function, preserving the input shape."""
    result = np.empty_like(x, dtype=float)
    positive = x >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    result[~positive] = exp_x / (1.0 + exp_x)
    return result


def mlp_loss_grad(parameters, x, y):
    """Two-layer tanh MLP; examples are rows and the loss is a mean."""
    w1, b1, w2, b2 = parameters
    hidden = np.tanh(x @ w1 + b1)
    logits = hidden @ w2 + b2
    loss = np.mean(np.logaddexp(0.0, logits) - y * logits)
    error = (sigmoid(logits) - y) / len(x)
    grad_w2 = hidden.T @ error
    grad_b2 = error.sum(axis=0)
    hidden_error = (error @ w2.T) * (1.0 - hidden * hidden)
    grad_w1 = x.T @ hidden_error
    grad_b1 = hidden_error.sum(axis=0)
    return float(loss), [grad_w1, grad_b1, grad_w2, grad_b2]


def gradient_check(parameters, x, y, epsilon=1e-5):
    """Check every parameter, restoring it after each perturbation."""
    _, analytic = mlp_loss_grad(parameters, x, y)
    worst = 0.0
    for parameter, derivative in zip(parameters, analytic):
        for index in np.ndindex(parameter.shape):
            old = parameter[index]
            parameter[index] = old + epsilon
            plus, _ = mlp_loss_grad(parameters, x, y)
            parameter[index] = old - epsilon
            minus, _ = mlp_loss_grad(parameters, x, y)
            parameter[index] = old
            numerical = (plus - minus) / (2.0 * epsilon)
            scale = max(1.0, abs(numerical), abs(derivative[index]))
            worst = max(worst, abs(numerical - derivative[index]) / scale)
    return worst


def attention(query, key, value, allowed=None):
    """Single scaled dot-product head with an optional Boolean mask."""
    scores = query @ key.T / math.sqrt(query.shape[-1])
    if allowed is not None:
        allowed = np.asarray(allowed, dtype=bool)
        if allowed.shape != scores.shape or not allowed.any(axis=1).all():
            raise ValueError("Each query needs at least one allowed key.")
        scores = np.where(allowed, scores, -np.inf)
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    return weights @ value, weights


def gcn_step(adjacency, features, weights):
    """One linear GCN layer for an undirected, unweighted graph."""
    adjacency = np.asarray(adjacency, dtype=float)
    with_loops = adjacency + np.eye(adjacency.shape[0])
    inverse_sqrt_degree = 1.0 / np.sqrt(with_loops.sum(axis=1))
    normalized = (inverse_sqrt_degree[:, None] * with_loops
                  * inverse_sqrt_degree[None, :])
    return normalized @ features @ weights


def diagonal_gaussian_kl(mean, std):
    """KL of a diagonal Gaussian to a standard Gaussian."""
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    if np.any(std <= 0):
        raise ValueError("Standard deviations must be positive.")
    variance = std * std
    return float(0.5 * np.sum(mean * mean + variance - 1 - np.log(variance)))


def main():
    rng = np.random.default_rng(7)
    x = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = np.array([[0.], [1.], [1.], [0.]])
    parameters = [rng.normal(0, 0.5, (2, 6)), np.zeros(6),
                  rng.normal(0, 0.5, (6, 1)), np.zeros(1)]
    gradient_error = gradient_check(parameters, x, y)
    assert gradient_error < 1e-8, gradient_error
    initial_loss, _ = mlp_loss_grad(parameters, x, y)
    for _ in range(4000):
        _, gradients = mlp_loss_grad(parameters, x, y)
        for parameter, gradient in zip(parameters, gradients):
            parameter -= 0.3 * gradient
    final_loss, _ = mlp_loss_grad(parameters, x, y)
    w1, b1, w2, b2 = parameters
    probabilities = sigmoid(np.tanh(x @ w1 + b1) @ w2 + b2)
    assert np.array_equal(probabilities >= 0.5, y.astype(bool))
    assert final_loss < 0.01

    # Chapter 16: all gradients are evaluated before the simultaneous update.
    w, b, v, c = 0.5, 0.0, 3.0, -1.0
    w, b, v, c = w - 0.01 * 6, b - 0.01 * 3, v - 0.01, c - 0.01
    prediction = v * max(0, w * 2 + b) + c
    assert np.isclose(prediction, 1.5315)
    scalar_loss = 0.5 * (prediction - 1) ** 2

    # Chapter 19: keys set addressing weights; values set delivered content.
    q = np.array([[1., 0.]])
    k = np.eye(2)
    values = np.array([[10.], [2.]])
    result, weights = attention(q, k, values)
    assert np.allclose(weights.sum(axis=1), 1)
    assert np.isclose(result.item(), 7.358092394613255)
    masked_result, masked_weights = attention(q, k, values, [[True, False]])
    assert masked_result.item() == 10
    assert np.array_equal(masked_weights, [[1, 0]])
    # Masked future values must have no effect on a causal output.
    causal = np.tril(np.ones((3, 3), dtype=bool))
    q3 = rng.normal(size=(3, 2))
    k3 = rng.normal(size=(3, 2))
    v3 = rng.normal(size=(3, 2))
    causal_output, _ = attention(q3, k3, v3, causal)
    altered = v3.copy()
    altered[-1] += 1000
    altered_output, _ = attention(q3, k3, altered, causal)
    assert np.allclose(causal_output[:-1], altered_output[:-1])

    # Chapter 20: normalization and equivariance under consistent relabeling.
    adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    features = np.array([[1.], [0.], [3.]])
    transformed = gcn_step(adjacency, features, np.ones((1, 1)))
    assert np.allclose(transformed[:, 0], [0.5, 4 / math.sqrt(6), 1.5])
    permutation = np.array([2, 0, 1])
    reordered = gcn_step(adjacency[np.ix_(permutation, permutation)],
                         features[permutation], np.ones((1, 1)))
    assert np.allclose(reordered, transformed[permutation])

    # Chapters 21, 22, and 24: exact constructed examples from the prose.
    kl = diagonal_gaussian_kl([1.0], [0.5])
    assert np.isclose(kl, 0.8181471805599453)
    assert diagonal_gaussian_kl([0., 0.], [1., 1.]) == 0
    scores = np.array([0.8, 0.4, 0.0]) / 0.2
    contrastive_loss = np.log(np.exp(scores - scores.max()).sum())
    assert np.isclose(contrastive_loss, 0.1429316284998995)
    perplexity = np.exp(-np.log([0.5, 0.5, 0.25, 0.5]).mean())
    assert np.isclose(perplexity, 32 ** 0.25)

    print(f"Maximum scaled gradient-check error: {gradient_error:.3e}")
    print(f"Four-point XOR loss: {initial_loss:.6f} -> {final_loss:.6f}")
    print("Four-point XOR probabilities:", np.round(probabilities[:, 0], 6))
    print(f"Chapter 16 scalar loss after update: {scalar_loss:.8f}")
    print(f"Chapter 19 attention output: {result.item():.6f}")
    print("Chapter 20 graph outputs:", np.round(transformed[:, 0], 6))
    print(f"Chapter 21 Gaussian KL: {kl:.6f}")
    print(f"Chapter 22 contrastive loss: {contrastive_loss:.6f}")
    print(f"Chapter 24 perplexity: {perplexity:.6f}")
    print("All mathematical and invariance checks passed.")


if __name__ == "__main__":
    main()
# End of companion_deep.py
