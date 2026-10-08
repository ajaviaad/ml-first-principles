"""Small, inspectable NumPy experiments for Chapters 16–27.

All data are generated locally. These examples expose mechanisms rather than
reproduce named, large-scale research systems. Examples use row-major batches.
"""
from __future__ import annotations

import math
import numpy as np
from book_code.companion_deep import (
    attention, diagonal_gaussian_kl, gcn_step, gradient_check,
    mlp_loss_grad, sigmoid,
)


def softmax(logits):
    """Normalize the last axis, subtracting its maximum for stability."""
    logits = np.asarray(logits, dtype=float)
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def conv2d(image, kernel, stride=1, padding=0):
    """Single-channel 2-D cross-correlation; no kernel reversal."""
    image, kernel = np.asarray(image, float), np.asarray(kernel, float)
    if image.ndim != 2 or kernel.ndim != 2 or stride < 1 or padding < 0:
        raise ValueError("Use two 2-D arrays, positive stride, nonnegative padding.")
    image = np.pad(image, padding)
    kh, kw = kernel.shape
    oh, ow = (image.shape[0] - kh) // stride + 1, (image.shape[1] - kw) // stride + 1
    if min(oh, ow) < 1:
        raise ValueError("Kernel is larger than the padded image.")
    result = np.empty((oh, ow))
    for row in range(oh):
        for col in range(ow):
            patch = image[row * stride:row * stride + kh, col * stride:col * stride + kw]
            result[row, col] = np.sum(patch * kernel)
    return result


def rnn_sequence(x, wx, wh, bias, initial=None):
    """Forward tanh RNN: x[T,D], wx[D,H], wh[H,H], output[T,H]."""
    state = np.zeros(wh.shape[0]) if initial is None else np.asarray(initial).copy()
    states = []
    for item in x:
        state = np.tanh(item @ wx + state @ wh + bias)
        states.append(state.copy())
    return np.asarray(states)


def lstm_step(x, hidden, cell, weights, bias):
    """One LSTM step. Packed gate order is forget, input, output, candidate."""
    gates = np.concatenate([x, hidden]) @ weights + bias
    f_raw, i_raw, o_raw, g_raw = np.split(gates, 4)
    f, i, o, candidate = sigmoid(f_raw), sigmoid(i_raw), sigmoid(o_raw), np.tanh(g_raw)
    new_cell = f * cell + i * candidate
    return o * np.tanh(new_cell), new_cell


def vae_loss_grad(parameters, x, epsilon, beta=1.0):
    """Linear VAE, unit observation variance, one latent, shared log variance.

    Return a one-sample Monte Carlo negative ELBO (without the Gaussian
    likelihood constant) and exact pathwise derivatives for FIXED epsilon.
    beta != 1 changes the objective away from the ordinary negative ELBO.
    Parameters: encoder[D,1], bias[1], logvar[1], decoder[1,D], bias[D].
    """
    ew, eb, logvar, dw, db = parameters
    mean = x @ ew + eb
    std = np.exp(0.5 * logvar)
    latent = mean + std * epsilon
    prediction = latent @ dw + db
    reconstruction = 0.5 * np.mean(np.sum((prediction - x) ** 2, axis=1))
    kl = 0.5 * np.mean(np.sum(mean ** 2 + np.exp(logvar) - 1 - logvar, axis=1))
    dp = (prediction - x) / len(x)
    gdw, gdb = latent.T @ dp, dp.sum(axis=0)
    dz = dp @ dw.T
    dm = dz + beta * mean / len(x)
    gell = np.sum(dz * epsilon * 0.5 * std, axis=0) + beta * 0.5 * (np.exp(logvar) - 1)
    return float(reconstruction + beta * kl), [x.T @ dm, dm.sum(axis=0), gell, gdw, gdb]


def info_nce(scores, temperature=1.0):
    """Rows are anchors; the matching item is the same column index.

    Return mean contrastive cross-entropy and its gradient with respect to
    the unscaled scores, not with respect to embeddings.
    """
    scores = np.asarray(scores, float)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1] or temperature <= 0:
        raise ValueError("Use square paired scores and positive temperature.")
    logits = scores / temperature
    maximum = logits.max(axis=1, keepdims=True)
    logsum = maximum[:, 0] + np.log(np.exp(logits - maximum).sum(axis=1))
    loss = np.mean(logsum - np.diag(logits))
    gradient = (softmax(logits) - np.eye(len(scores))) / (len(scores) * temperature)
    return float(loss), gradient


def normalize_rows(x):
    x = np.asarray(x, float)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Cosine similarity is undefined for zero vectors.")
    return x / norms


def lora_apply(x, weight, a, b, scale=1.0):
    """Apply x(W + scale AB) without materializing the rank-r update."""
    return x @ weight + scale * (x @ a) @ b


def fit_bigram(sequences, vocabulary, smoothing=0.5):
    """Additive-smoothed categorical transitions with declared vocabulary."""
    if smoothing <= 0 or len(set(vocabulary)) != len(vocabulary):
        raise ValueError("Use positive smoothing and a unique vocabulary.")
    index = {word: i for i, word in enumerate(vocabulary)}
    counts = np.full((len(index), len(index)), smoothing)
    for sequence in sequences:
        for left, right in zip(sequence[:-1], sequence[1:]):
            counts[index[left], index[right]] += 1
    return counts / counts.sum(axis=1, keepdims=True)


def perplexity(probabilities):
    values = np.asarray(probabilities, float)
    if values.size == 0 or np.any((values <= 0) | (values > 1)):
        raise ValueError("Perplexity requires nonempty probabilities in (0, 1].")
    return float(np.exp(-np.log(values).mean()))


def gan_losses_grad(real, noise, mean, slope, intercept):
    """Location-only generator G(z)=z+mean; logistic discriminator.

    D minimizes real/fake BCE; G minimizes non-saturating fake BCE.
    The return gradients hold the other player fixed.
    """
    fake = noise + mean
    real_logits, fake_logits = slope * real + intercept, slope * fake + intercept
    dr, df = sigmoid(real_logits), sigmoid(fake_logits)
    dloss = np.mean(np.logaddexp(0, -real_logits) + np.logaddexp(0, fake_logits))
    gloss = np.mean(np.logaddexp(0, -fake_logits))
    da = np.mean((dr - 1) * real + df * fake)
    db = np.mean(dr - 1 + df)
    dm = np.mean((df - 1) * slope)
    return float(dloss), float(gloss), float(da), float(db), float(dm)


def diffusion_forward(clean, alpha_bar, noise):
    if not 0 < alpha_bar <= 1:
        raise ValueError("Cumulative alpha must be in (0, 1].")
    return np.sqrt(alpha_bar) * np.asarray(clean) + np.sqrt(1 - alpha_bar) * np.asarray(noise)


def gaussian_reverse_step(noisy, alpha, previous_alpha_bar, data_mean, data_variance, noise):
    """Exact Gaussian reverse conditional using an ORACLE data distribution.

    This does not know individual clean samples. The analytic denoiser knows
    the population mean/variance. It is a sampler identity, not a trained DDPM.
    """
    if not 0 < alpha < 1 or not 0 < previous_alpha_bar <= 1 or data_variance <= 0:
        raise ValueError("Invalid Gaussian reverse-process parameters.")
    previous_mean = np.sqrt(previous_alpha_bar) * data_mean
    previous_var = previous_alpha_bar * data_variance + 1 - previous_alpha_bar
    current_mean = np.sqrt(alpha) * previous_mean
    current_var = alpha * previous_var + 1 - alpha
    coefficient = np.sqrt(alpha) * previous_var / current_var
    conditional_mean = previous_mean + coefficient * (np.asarray(noisy) - current_mean)
    conditional_var = max(0.0, previous_var - alpha * previous_var ** 2 / current_var)
    return conditional_mean + np.sqrt(conditional_var) * noise


def orthogonal_alignment(source, target):
    """Fit an orthogonal map from paired rows by solving Procrustes with SVD."""
    u, _, vt = np.linalg.svd(np.asarray(source).T @ np.asarray(target))
    return u @ vt


def lesson_16(seed=42):
    rng = np.random.default_rng(seed)
    x = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = np.array([[0.], [1.], [1.], [0.]])
    p = [rng.normal(0, .5, (2, 8)), np.zeros(8), rng.normal(0, .5, (8, 1)), np.zeros(1)]
    error = gradient_check(p, x, y)
    initial, _ = mlp_loss_grad(p, x, y)
    for _ in range(2500):
        _, gradients = mlp_loss_grad(p, x, y)
        for parameter, gradient in zip(p, gradients):
            parameter -= .3 * gradient
    final, _ = mlp_loss_grad(p, x, y)
    probabilities = sigmoid(np.tanh(x @ p[0] + p[1]) @ p[2] + p[3])
    return {"initial_loss": initial, "final_loss": final, "gradient_error": float(error),
            "xor_probabilities": probabilities[:, 0].tolist(),
            "training_accuracy": float(np.mean((probabilities >= .5) == y)),
            "scalar_update_prediction": 1.5315, "scope": "Four-point training fit, not generalization."}


def lesson_17(seed=42):
    image = np.tile(np.array([1., 1., 4., 4.]), (4, 1))
    edges = conv2d(image, np.array([[-1., 1.]]))
    residual = image + conv2d(image, np.zeros((3, 3)), padding=1)
    return {"input_shape": list(image.shape), "feature_shape": list(edges.shape),
            "edge_response": edges[0].tolist(), "global_average": float(edges.mean()),
            "zero_branch_residual_error": float(np.max(np.abs(image - residual))),
            "dense_convolution_weights": 3 * 3 * 64 * 64,
            "depthwise_separable_weights": 3 * 3 * 64 + 64 * 64,
            "scope": "Fixed-filter forward operations; no trained image classifier."}


def lesson_18(seed=42):
    x = np.array([[1.], [0.], [0.], [0.]])
    states = rnn_sequence(x, np.ones((1, 1)), np.array([[.8]]), np.zeros(1))
    # Zero gate weights set explicit, constant gates via their logits.
    bias = np.array([np.log(.9 / .1), np.log(.2 / .8), 0., np.arctanh(-.5)])
    hidden, cell = lstm_step(np.array([0.]), np.array([0.]), np.array([2.]), np.zeros((2, 4)), bias)
    return {"rnn_states": states[:, 0].tolist(), "lstm_cell": float(cell[0]),
            "lstm_hidden": float(hidden[0]), "contracting_path_50": .8 ** 50,
            "expanding_path_50": 1.2 ** 50, "forget_path_50": .99 ** 50,
            "scope": "Forward recurrence and direct-path derivatives; no BPTT training."}


def lesson_19(seed=42):
    q, k, v = np.array([[1., 0.]]), np.eye(2), np.array([[10.], [2.]])
    result, weights = attention(q, k, v)
    rng = np.random.default_rng(seed)
    q3, k3, v3 = (rng.normal(size=(3, 2)) for _ in range(3))
    causal = np.tril(np.ones((3, 3), dtype=bool))
    original, _ = attention(q3, k3, v3, causal)
    modified = v3.copy()
    modified[-1] += 1000
    changed, _ = attention(q3, k3, modified, causal)
    return {"weights": weights.tolist(), "weighted_value": float(result[0, 0]),
            "causal_prefix_error": float(np.max(np.abs(original[:-1] - changed[:-1]))),
            "attention_score_shape": [3, 3], "scope": "One attention head, not a complete transformer."}


def lesson_20(seed=42):
    a = np.array([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
    x = np.array([[1.], [0.], [3.]])
    w = np.ones((1, 1))
    result = gcn_step(a, x, w)
    order = np.array([2, 0, 1])
    reordered = gcn_step(a[np.ix_(order, order)], x[order], w)
    smooth = x.copy()
    for _ in range(40):
        smooth = gcn_step(a, smooth, w)
    # Symmetric normalization converges to degree-weighted, not constant states.
    degree = (a + np.eye(3)).sum(axis=1)
    scaled = smooth[:, 0] / np.sqrt(degree)
    return {"node_outputs": result[:, 0].tolist(),
            "permutation_error": float(np.max(np.abs(reordered - result[order]))),
            "degree_adjusted_spread_after_40": float(np.ptp(scaled)),
            "scope": "Fixed linear propagation on an undirected path; no learned graph task."}


def lesson_21(seed=42):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=(120, 1))
    x = np.hstack([signal, 2 * signal]) + .08 * rng.normal(size=(120, 2))
    center = x.mean(axis=0)
    centered = x - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    codes = centered @ vt[:1].T
    reconstruction = codes @ vt[:1] + center
    p = [rng.normal(0, .1, (2, 1)), np.zeros(1), np.zeros(1), rng.normal(0, .1, (1, 2)), np.zeros(2)]
    evaluation_noise = rng.normal(size=(len(x), 1))
    initial, _ = vae_loss_grad(p, x, evaluation_noise)
    for _ in range(600):
        _, grads = vae_loss_grad(p, x, rng.normal(size=(len(x), 1)))
        for parameter, gradient in zip(p, grads):
            parameter -= .025 * gradient
    final, _ = vae_loss_grad(p, x, evaluation_noise)
    return {"linear_autoencoder_mse": float(np.mean((reconstruction - x) ** 2)),
            "mean_only_mse": float(np.mean(centered ** 2)),
            "vae_initial_negative_elbo_without_constant": initial,
            "vae_final_negative_elbo_without_constant": final,
            "vae_posterior_std": float(np.exp(.5 * p[2][0])),
            "gaussian_kl_example": diagonal_gaussian_kl([1.], [.5]),
            "reparameterized_sample": 1 + .5 * -2,
            "scope": "SVD linear AE and trained linear one-latent VAE; unit observation variance."}


def lesson_22(seed=42):
    rng = np.random.default_rng(seed)
    first = normalize_rows(rng.normal(size=(12, 8)))
    second = normalize_rows(first + .03 * rng.normal(size=first.shape))
    scores = first @ second.T
    paired_loss, _ = info_nce(scores, .2)
    shuffled_loss, _ = info_nce(scores[:, np.roll(np.arange(12), 1)], .2)
    collapsed_loss, _ = info_nce(np.ones((12, 12)), .2)
    hand_logits = np.array([.8, .4, 0.]) / .2
    hand_loss = float(np.log(np.exp(hand_logits - hand_logits[0]).sum()))
    return {"paired_info_nce": paired_loss, "shuffled_pair_info_nce": shuffled_loss,
            "collapsed_info_nce": collapsed_loss, "collapse_reference_log_batch": math.log(12),
            "pair_retrieval_accuracy": float(np.mean(scores.argmax(axis=1) == np.arange(12))),
            "hand_example_loss": hand_loss,
            "scope": "Contrastive objective and score gradients on constructed views; no encoder training."}


def lesson_23(seed=42):
    rng = np.random.default_rng(seed)
    train = rng.uniform(-1, 1, (24, 2))
    test = rng.uniform(-1, 1, (400, 2))
    y = np.where(train[:, 0] * train[:, 1] > 0, 1., -1.)
    target = test[:, 0] * test[:, 1] > 0
    def features(x):
        return np.column_stack([np.ones(len(x)), x[:, 0] * x[:, 1]])
    f = features(train)
    head = np.linalg.solve(f.T @ f + .01 * np.eye(2), f.T @ y)
    raw = np.column_stack([np.ones(len(train)), train])
    raw_head = np.linalg.solve(raw.T @ raw + .01 * np.eye(3), raw.T @ y)
    x, w = rng.normal(size=(6, 8)), rng.normal(size=(8, 5))
    a, b = rng.normal(size=(8, 2)), rng.normal(size=(2, 5))
    factorized = lora_apply(x, w, a, b, .5)
    return {"labeled_training_examples": len(train), "heldout_examples": len(test),
            "frozen_feature_accuracy": float(np.mean((features(test) @ head > 0) == target)),
            "raw_linear_accuracy": float(np.mean((np.column_stack([np.ones(len(test)), test]) @ raw_head > 0) == target)),
            "lora_materialization_error": float(np.max(np.abs(factorized - x @ (w + .5 * a @ b)))),
            "lora_trainable_parameters": int(a.size + b.size), "dense_parameters": int(w.size),
            "scope": "Hand-designed frozen features and exact low-rank update algebra; no pretrained model."}


def lesson_24(seed=42):
    vocabulary = ['<bos>', 'the', 'red', 'blue', 'cat', 'dog', 'sleeps', 'runs', '<eos>']
    sentences = ['the red cat sleeps', 'the blue dog runs', 'the red dog sleeps', 'the blue cat runs']
    train = [['<bos>', *sentence.split(), '<eos>'] for sentence in sentences]
    # New compositions are evaluation examples, never counted in fitting.
    heldout = [['<bos>', *s.split(), '<eos>'] for s in ['the red cat runs', 'the blue dog sleeps']]
    probabilities = fit_bigram(train, vocabulary)
    index = {word: i for i, word in enumerate(vocabulary)}
    observed = [probabilities[index[a], index[b]] for row in heldout for a, b in zip(row[:-1], row[1:])]
    rng = np.random.default_rng(seed)
    current = index['<bos>']
    generated = []
    for _ in range(20):
        current = int(rng.choice(len(vocabulary), p=probabilities[current]))
        if vocabulary[current] == '<eos>':
            break
        generated.append(vocabulary[current])
    return {"vocabulary_size": len(vocabulary), "heldout_perplexity": perplexity(observed),
            "uniform_perplexity": float(len(vocabulary)), "hand_perplexity": perplexity([.5, .5, .25, .5]),
            "generated_tokens": generated, "row_sum_error": float(np.max(np.abs(probabilities.sum(axis=1) - 1))),
            "scope": "Smoothed word bigram, including special tokens in all output rows; not an LLM."}


def lesson_25(seed=42):
    rng = np.random.default_rng(seed)
    mean, slope, intercept = -1., .1, 0.
    initial = mean
    for _ in range(1400):
        real, noise = rng.normal(2., 1., 128), rng.normal(size=128)
        _, _, da, db, _ = gan_losses_grad(real, noise, mean, slope, intercept)
        slope, intercept = slope - .08 * da, intercept - .08 * db
        _, _, _, _, dm = gan_losses_grad(real, noise, mean, slope, intercept)
        mean -= .08 * dm
    real, noise = rng.normal(2., 1., 5000), rng.normal(size=5000)
    dloss, gloss, *_ = gan_losses_grad(real, noise, mean, slope, intercept)
    return {"initial_generator_mean": initial, "learned_generator_mean": float(mean),
            "target_mean": 2., "generated_sample_mean": float(np.mean(noise + mean)),
            "generated_sample_std": float(np.std(noise + mean)), "discriminator_loss": dloss,
            "generator_loss": gloss,
            "scope": "Location-only Gaussian GAN; fixed variance and unimodal generator limit coverage."}


def lesson_26(seed=42):
    rng = np.random.default_rng(seed)
    alphas = 1 - np.linspace(.04, .12, 60)
    bars = np.concatenate([[1.], np.cumprod(alphas)])
    mean, variance = 2., .25
    terminal_mean = np.sqrt(bars[-1]) * mean
    terminal_variance = bars[-1] * variance + 1 - bars[-1]
    current = rng.normal(terminal_mean, np.sqrt(terminal_variance), 12000)
    for t in range(len(alphas), 0, -1):
        current = gaussian_reverse_step(current, alphas[t - 1], bars[t - 1], mean, variance,
                                        rng.normal(size=current.shape))
    noisy = float(diffusion_forward(.8, .64, -.5))
    return {"forward_example": noisy, "recovered_with_known_noise": (noisy - .6 * -.5) / .8,
            "terminal_alpha_bar": float(bars[-1]), "reverse_sample_mean": float(current.mean()),
            "reverse_sample_std": float(current.std()), "target_mean": mean, "target_std": math.sqrt(variance),
            "reverse_steps": len(alphas),
            "scope": "Exact Gaussian oracle reverse conditionals; initialized at exact terminal marginal; no denoiser training."}


def lesson_27(seed=42):
    rng = np.random.default_rng(seed)
    rotation, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    training_images = rng.normal(size=(50, 6))
    training_text = training_images @ rotation + .02 * rng.normal(size=(50, 6))
    mapping = orthogonal_alignment(training_images, training_text)
    images = rng.normal(size=(30, 6))
    text = images @ rotation + .02 * rng.normal(size=(30, 6))
    raw_scores = normalize_rows(images) @ normalize_rows(text).T
    aligned_scores = normalize_rows(images @ mapping) @ normalize_rows(text).T
    inv_var = 1 / np.array([20. ** 2, 10. ** 2])
    weights = inv_var / inv_var.sum()
    return {"heldout_pairs": len(images),
            "retrieval_before_alignment": float(np.mean(raw_scores.argmax(axis=1) == np.arange(len(images)))),
            "retrieval_after_alignment": float(np.mean(aligned_scores.argmax(axis=1) == np.arange(len(images)))),
            "fusion_weights": weights.tolist(), "fused_speed": float(weights @ np.array([1000., 1060.])),
            "fused_std": float(np.sqrt(1 / inv_var.sum())), "aligned_audio_time": 1.2 + .4,
            "scope": "Synthetic paired numeric modalities and orthogonal alignment; no image/text encoders."}
