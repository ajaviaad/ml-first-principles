"""Independent arithmetic, finite-difference, and invariance checks."""
import json
import math
import unittest
import numpy as np
from mlfirst import neural as n


class NeuralTests(unittest.TestCase):
    def test_mlp_gradients_and_xor_fit(self):
        result = n.lesson_16()
        self.assertLess(result['gradient_error'], 1e-8)
        self.assertLess(result['final_loss'], .02)
        self.assertEqual(result['training_accuracy'], 1.)

    def test_convolution_hand_calculation_and_shape(self):
        result = n.conv2d([[1, 1, 4, 4]], [[-1, 1]])
        np.testing.assert_array_equal(result, [[0, 3, 0]])
        self.assertEqual(n.conv2d(np.zeros((32, 32)), np.ones((3, 3)), 2, 1).shape, (16, 16))
        with self.assertRaises(ValueError):
            n.conv2d(np.ones((2, 2)), np.ones((3, 3)))

    def test_recurrent_chunking_preserves_state(self):
        x = np.array([[.1], [.2], [-.1], [.3]])
        args = (np.array([[.7]]), np.array([[.4]]), np.zeros(1))
        full = n.rnn_sequence(x, *args)
        first = n.rnn_sequence(x[:2], *args)
        second = n.rnn_sequence(x[2:], *args, initial=first[-1])
        np.testing.assert_allclose(full, np.vstack([first, second]))
        expected = np.tanh(.1 * .7)
        self.assertAlmostEqual(full[0, 0], expected)

    def test_lstm_gates(self):
        result = n.lesson_18()
        self.assertAlmostEqual(result['lstm_cell'], 1.7)
        self.assertAlmostEqual(result['lstm_hidden'], .5 * math.tanh(1.7))

    def test_attention_mask_and_causal_independence(self):
        q, k, v = np.array([[1., 0.]]), np.eye(2), np.array([[10.], [2.]])
        output, weights = n.attention(q, k, v, [[True, False]])
        np.testing.assert_array_equal(weights, [[1., 0.]])
        self.assertEqual(output.item(), 10.)
        with self.assertRaises(ValueError):
            n.attention(q, k, v, [[False, False]])
        self.assertEqual(n.lesson_19()['causal_prefix_error'], 0.)

    def test_graph_normalization_and_permutation(self):
        result = n.lesson_20()
        np.testing.assert_allclose(result['node_outputs'], [.5, 4 / math.sqrt(6), 1.5])
        self.assertLess(result['permutation_error'], 1e-14)
        self.assertLess(result['degree_adjusted_spread_after_40'], 1e-10)

    def test_vae_pathwise_gradients_with_fixed_noise(self):
        rng = np.random.default_rng(13)
        x, epsilon = rng.normal(size=(4, 2)), rng.normal(size=(4, 1))
        p = [rng.normal(size=(2, 1)) * .2, np.array([.1]), np.array([-.3]),
             rng.normal(size=(1, 2)) * .2, np.array([.1, -.1])]
        _, gradients = n.vae_loss_grad(p, x, epsilon, beta=.7)
        for parameter, gradient in zip(p, gradients):
            for index in np.ndindex(parameter.shape):
                old = parameter[index]
                parameter[index] = old + 1e-5
                plus, _ = n.vae_loss_grad(p, x, epsilon, beta=.7)
                parameter[index] = old - 1e-5
                minus, _ = n.vae_loss_grad(p, x, epsilon, beta=.7)
                parameter[index] = old
                self.assertAlmostEqual(gradient[index], (plus - minus) / 2e-5, places=7)
        self.assertAlmostEqual(n.diagonal_gaussian_kl([1.], [.5]), .8181471805599453)
        self.assertEqual(n.diagonal_gaussian_kl([0.], [1.]), 0.)

    def test_info_nce_gradient_and_collapse(self):
        scores = np.array([[.7, -.1], [.3, .9]])
        _, gradient = n.info_nce(scores, .4)
        for index in np.ndindex(scores.shape):
            perturbation = np.zeros_like(scores)
            perturbation[index] = 1e-5
            plus, _ = n.info_nce(scores + perturbation, .4)
            minus, _ = n.info_nce(scores - perturbation, .4)
            self.assertAlmostEqual(gradient[index], (plus - minus) / 2e-5, places=7)
        self.assertAlmostEqual(n.info_nce(np.ones((5, 5)), .2)[0], math.log(5))
        with self.assertRaises(ValueError):
            n.info_nce(scores, 0)

    def test_lora_exact_factorization_and_rank(self):
        rng = np.random.default_rng(2)
        x, w, a, b = (rng.normal(size=shape) for shape in [(3, 7), (7, 5), (7, 2), (2, 5)])
        np.testing.assert_allclose(n.lora_apply(x, w, a, b, .3), x @ (w + .3 * a @ b))
        self.assertLessEqual(np.linalg.matrix_rank(a @ b), 2)

    def test_bigram_counts_and_perplexity(self):
        probabilities = n.fit_bigram([['a', 'b', 'a']], ['a', 'b'], 1.)
        np.testing.assert_allclose(probabilities, [[1 / 3, 2 / 3], [2 / 3, 1 / 3]])
        self.assertAlmostEqual(n.perplexity([.5, .5, .25, .5]), 32 ** .25)
        self.assertAlmostEqual(n.perplexity([.25] * 4), 4.)
        with self.assertRaises(ValueError):
            n.perplexity([0.])

    def test_gan_player_gradients(self):
        real, noise = np.array([1., 2., 4.]), np.array([-1., 0., 1.])
        mean, slope, intercept, eps = .3, .8, -.1, 1e-5
        dloss, gloss, da, db, dm = n.gan_losses_grad(real, noise, mean, slope, intercept)
        self.assertGreater(dloss, 0.)
        self.assertGreater(gloss, 0.)
        numerical_da = (n.gan_losses_grad(real, noise, mean, slope + eps, intercept)[0] -
                        n.gan_losses_grad(real, noise, mean, slope - eps, intercept)[0]) / (2 * eps)
        numerical_db = (n.gan_losses_grad(real, noise, mean, slope, intercept + eps)[0] -
                        n.gan_losses_grad(real, noise, mean, slope, intercept - eps)[0]) / (2 * eps)
        numerical_dm = (n.gan_losses_grad(real, noise, mean + eps, slope, intercept)[1] -
                        n.gan_losses_grad(real, noise, mean - eps, slope, intercept)[1]) / (2 * eps)
        np.testing.assert_allclose([da, db, dm], [numerical_da, numerical_db, numerical_dm], atol=1e-9)

    def test_diffusion_forward_and_reverse_moments(self):
        self.assertAlmostEqual(float(n.diffusion_forward(.8, .64, -.5)), .34)
        # A known marginal transformed by its exact reverse conditional must
        # recover the previous marginal, checked by an independent sample.
        rng = np.random.default_rng(3)
        alpha, previous_bar, mean, variance = .8, .5, 2., .25
        current_bar = alpha * previous_bar
        current = rng.normal(np.sqrt(current_bar) * mean,
                             np.sqrt(current_bar * variance + 1 - current_bar), 80000)
        previous = n.gaussian_reverse_step(current, alpha, previous_bar, mean, variance,
                                          rng.normal(size=current.shape))
        self.assertAlmostEqual(previous.mean(), np.sqrt(previous_bar) * mean, delta=.012)
        self.assertAlmostEqual(previous.var(), previous_bar * variance + 1 - previous_bar, delta=.015)

    def test_alignment_fits_unseen_pairs(self):
        rng = np.random.default_rng(9)
        rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        x = rng.normal(size=(15, 4))
        mapping = n.orthogonal_alignment(x, x @ rotation)
        np.testing.assert_allclose(mapping, rotation, atol=1e-13)
        np.testing.assert_allclose(mapping.T @ mapping, np.eye(4), atol=1e-13)
        result = n.lesson_27()
        self.assertEqual(result['retrieval_after_alignment'], 1.)
        self.assertAlmostEqual(result['fused_speed'], 1048.)
        self.assertAlmostEqual(result['fused_std'], math.sqrt(80))

    def test_every_lesson_serializes_finite_values(self):
        for chapter in range(16, 28):
            with self.subTest(chapter=chapter):
                result = getattr(n, f'lesson_{chapter}')()
                json.dumps(result, allow_nan=False)
                self.assertIn('scope', result)


if __name__ == '__main__':
    unittest.main()
