"""Independent arithmetic, finite differences, exhaustive checks and invariants."""
import itertools
import json
import unittest
import numpy as np
from mlfirst import classical as c


class ClassicalTests(unittest.TestCase):
    def test_least_squares_closed_form_and_free_intercept(self):
        coefficient = c.ridge_fit([[0], [1], [2]], [1, 2, 2])
        np.testing.assert_allclose(coefficient, [7 / 6, 1 / 2])
        np.testing.assert_allclose(c.ridge_fit([[0], [1], [2]], [1, 2, 2], 2), [17 / 12, 1 / 4])
        np.testing.assert_allclose(c.ridge_fit(np.ones((4, 2)), np.full(4, 7), 100), [7, 0, 0], atol=1e-12)

    def test_polynomial_recovers_known_curve(self):
        x = np.array([-2, -1, 0, 1, 2.])
        X = c.polynomial_features(x, 2)
        model = c.ridge_fit(X, 3 - 2 * x + 0.5 * x ** 2)
        np.testing.assert_allclose(model, [3, -2, .5], atol=1e-12)

    def test_logistic_gradient_against_finite_differences(self):
        design = np.array([[1, -2, 1], [1, .5, -.5], [1, 3, 2.]])
        labels = np.array([0, 1, 1.])
        weight = np.array([.3, -.2, .7])
        _, analytic = c.logistic_loss_gradient(design, labels, weight, .13)
        numerical = []
        for coordinate in np.eye(3) * 1e-6:
            high = c.logistic_loss_gradient(design, labels, weight + coordinate, .13)[0]
            low = c.logistic_loss_gradient(design, labels, weight - coordinate, .13)[0]
            numerical.append((high - low) / 2e-6)
        np.testing.assert_allclose(analytic, numerical, atol=1e-8)
        np.testing.assert_allclose(c.sigmoid([-1000, 0, 1000]), [0, .5, 1])

    def test_logistic_descent_and_nb_presence_absence(self):
        model = c.logistic_fit([[-2], [-1], [1], [2]], [0, 0, 1, 1])
        self.assertLess(model['loss_history'][-1], model['loss_history'][0])
        self.assertTrue(np.all(np.diff(model['loss_history']) <= 1e-12))
        nb = c.bernoulli_nb_fit([[0], [0], [1], [1]], [0, 0, 1, 1], alpha=1)
        np.testing.assert_allclose(nb['feature_probability'], [[.25], [.75]])
        np.testing.assert_allclose(c.bernoulli_nb_predict_proba(nb, [[0], [1]]), [[.75, .25], [.25, .75]])
        archive = {'class_prior': [.75, .25], 'feature_probability': [[.1, .2], [.8, .6]]}
        self.assertAlmostEqual(c.bernoulli_nb_predict_proba(archive, [[1, 1]])[0, 1], 8 / 9)

    def test_neighbor_exact_arithmetic_and_zero_distance(self):
        self.assertAlmostEqual(c.knn_probability([[1], [2], [3]], [1, 0, 0], [0], 3), 1 / 3)
        self.assertAlmostEqual(c.knn_probability([[1], [2], [3]], [1, 0, 0], [0], 3, True), 6 / 11)
        self.assertAlmostEqual(c.knn_probability([[0], [0], [1]], [1, 0, 1], [0], 3, True), .5)
        self.assertEqual(c.knn_probability([[-1], [1]], [0, 1], [0], 1), 0)

    def test_hinge_gradient_and_kernel_properties(self):
        X, y, w, b = np.array([[0.], [2.], [3.]]), [-1, 1, 1], np.array([.3]), -.2
        loss, dw, db = c.hinge_loss_gradient(X, y, w, b, .2)
        epsilon = 1e-6
        numerical_w = (c.hinge_loss_gradient(X, y, w + epsilon, b, .2)[0] -
                       c.hinge_loss_gradient(X, y, w - epsilon, b, .2)[0]) / (2 * epsilon)
        numerical_b = (c.hinge_loss_gradient(X, y, w, b + epsilon, .2)[0] -
                       c.hinge_loss_gradient(X, y, w, b - epsilon, .2)[0]) / (2 * epsilon)
        np.testing.assert_allclose(dw, [numerical_w], atol=1e-9)
        self.assertAlmostEqual(db, numerical_b)
        kernel = c.rbf_kernel(X, X, .7)
        np.testing.assert_allclose(kernel, kernel.T)
        np.testing.assert_allclose(np.diag(kernel), np.ones(3))
        self.assertGreaterEqual(np.linalg.eigvalsh(kernel).min(), -1e-12)
        self.assertAlmostEqual(kernel[0, 1], np.exp(-2.8))

    def test_tree_closed_form_and_minimum_leaf(self):
        X, y = np.arange(1, 5)[:, None], [1, 2, 8, 9]
        np.testing.assert_allclose(c.regression_stump(X, y), [1, 0, 2.5, 1.5, 8.5])
        np.testing.assert_allclose(c.tree_predict(c.tree_fit(X, y, 2), X), y)
        np.testing.assert_allclose(c.tree_predict(c.tree_fit(X, y, 5, min_leaf=2), X), [1.5, 1.5, 8.5, 8.5])
        np.testing.assert_allclose(c.tree_predict(c.tree_fit(X, y, 0), X), np.full(4, 5))
        self.assertAlmostEqual(c.gini([1] * 6 + [0] * 4), .48)
        xor = [[0, 0], [0, 1], [1, 0], [1, 1]]
        self.assertNotIn('feature', c.tree_fit(xor, [0, 1, 1, 0], 2))

    def test_ensemble_residual_arithmetic_and_reproducibility(self):
        X, y = np.arange(1, 5)[:, None], [2, 2, 6, 6]
        boost = c.boosting_fit(X, y, 2, .5)
        np.testing.assert_allclose(boost['sse_history'], [16, 4, 1])
        np.testing.assert_allclose(c.boosting_predict(boost, X), [2.5, 2.5, 5.5, 5.5])
        a, b = c.bagging_fit(X, y, seed=9), c.bagging_fit(X, y, seed=9)
        np.testing.assert_array_equal(a['bootstrap_indices'], b['bootstrap_indices'])
        np.testing.assert_allclose(c.bagging_predict(a, X), c.bagging_predict(b, X))
        constant = c.boosting_fit(np.ones((4, 1)), y)
        self.assertEqual(len(constant['stumps']), 0)

    def test_split_separates_adjacent_representable_feature_values(self):
        low = np.nextafter(1., 2.)
        high = np.nextafter(low, 2.)
        # Their rounded arithmetic midpoint equals high, making <= midpoint
        # put both rows on the left unless the threshold has an explicit bound.
        self.assertEqual(low / 2 + high / 2, high)
        X = np.array([[low], [high]])
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            stump = c.regression_stump(X, [0, 1])
            tree = c.tree_fit(X, [0, 1], max_depth=1)
        self.assertEqual(stump[0], 0)
        self.assertGreaterEqual(stump[2], low)
        self.assertLess(stump[2], high)
        np.testing.assert_array_equal(c.stump_predict(stump, X), [0, 1])
        np.testing.assert_array_equal(c.tree_predict(tree, X), [0, 1])

    def test_split_separates_opposite_large_finite_feature_values(self):
        X = np.array([[-1e308], [1e308]])
        # Subtracting endpoints would overflow, but an exact partition and
        # ordinary small response means require no such subtraction.
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            stump = c.regression_stump(X, [0, 1])
            tree = c.tree_fit(X, [0, 1], max_depth=1)
        self.assertTrue(np.isfinite(stump).all())
        self.assertEqual(stump[0], 0)
        self.assertEqual(stump[2], 0)
        np.testing.assert_array_equal(c.stump_predict(stump, X), [0, 1])
        np.testing.assert_array_equal(c.tree_predict(tree, X), [0, 1])

    def test_kmeans_stationarity_and_objective(self):
        X = np.array([[0.], [1.], [4.], [9.]])
        labels, centers, trace = c.kmeans(X, [[0], [9]])
        np.testing.assert_array_equal(labels, [0, 0, 0, 1])
        np.testing.assert_allclose(centers[:, 0], [5 / 3, 9])
        self.assertAlmostEqual(trace[-1], 26 / 3)
        self.assertTrue(np.all(np.diff(trace) <= 1e-12))
        for group in range(len(centers)):
            np.testing.assert_allclose(centers[group], X[labels == group].mean(axis=0))
        with self.assertRaises(ValueError):
            c.kmeans(X, [[0], [0]])

    def test_pca_reconstruction_subspace_and_heldout_center(self):
        X = np.array([[2., 0., 1.], [-2., 0., -1.], [0., 1., 0.], [0., -1., 0.]])
        model = c.pca_fit(X, 1)
        reconstruction = c.pca_inverse_transform(model, c.pca_transform(model, X))
        self.assertAlmostEqual(np.sum((X - reconstruction) ** 2), 2)
        self.assertAlmostEqual(np.sum(model['singular_values'][1:] ** 2), 2)
        np.testing.assert_allclose(model['components'] @ model['components'].T, [[1]])
        shifted = c.pca_fit(X + 10, 3)
        new = np.array([[100, 100, 100.]])
        np.testing.assert_allclose(c.pca_inverse_transform(shifted, c.pca_transform(shifted, new)), new)
        np.testing.assert_allclose(c.pca_fit(np.ones((3, 2)), 1)['explained_variance_ratio'], [0, 0])

    def test_apriori_matches_exhaustive_support(self):
        baskets = [{'a', 'b'}, {'a', 'b', 'c'}, {'a', 'c'}, {'b'}, set()]
        expected = {}
        for size in range(1, 4):
            for subset in itertools.combinations('abc', size):
                key = frozenset(subset)
                count = sum(key <= basket for basket in baskets)
                if count >= 2:
                    expected[key] = count
        self.assertEqual(c.frequent_itemsets(baskets, 2), expected)
        rules = c.association_rules(baskets, 2, 0)
        rule = next(r for r in rules if r['antecedent'] == ['a'] and r['consequent'] == ['b'])
        self.assertAlmostEqual(rule['support'], 2 / 5)
        self.assertAlmostEqual(rule['confidence'], 2 / 3)
        self.assertAlmostEqual(rule['lift'], 10 / 9)
        self.assertEqual(c.association_rules([], 2), [])

    def test_hmm_against_complete_path_enumeration(self):
        initial = np.array([.9, .1])
        transition = np.array([[.95, .05], [.1, .9]])
        emission = np.array([[.9, .1], [.2, .8]])
        observations = [1, 1, 0]
        paths = list(itertools.product(range(2), repeat=3))
        probabilities = []
        for path in paths:
            p = initial[path[0]]
            for t, observation in enumerate(observations):
                p *= emission[path[t], observation]
                if t:
                    p *= transition[path[t - 1], path[t]]
            probabilities.append(p)
        total = sum(probabilities)
        filtered, log_evidence = c.hmm_forward(initial, transition, emission, observations)
        smoothed = c.hmm_smooth(initial, transition, emission, observations)
        self.assertAlmostEqual(np.exp(log_evidence), total)
        for t in range(3):
            marginal = sum(p for path, p in zip(paths, probabilities) if path[t] == 1) / total
            self.assertAlmostEqual(smoothed[t, 1], marginal)
        self.assertAlmostEqual(filtered[-1, 1], smoothed[-1, 1])
        path, log_joint = c.hmm_viterbi(initial, transition, emission, observations)
        self.assertEqual(path, list(paths[int(np.argmax(probabilities))]))
        self.assertAlmostEqual(np.exp(log_joint), max(probabilities))
        self.assertAlmostEqual(filtered[0, 1], 8 / 17)

    def test_hmm_empty_impossible_and_validation(self):
        for function in (c.hmm_forward, c.hmm_smooth, c.hmm_viterbi):
            function([1], [[1]], [[1, 0]], [])
            with self.assertRaises(ValueError):
                function([1], [[1]], [[1, 0]], [1])
            with self.assertRaises(ValueError):
                function([.8], [[1]], [[1, 0]], [0])
            with self.assertRaises(ValueError):
                function([1], [[1]], [[1, 0]], [-1])

    def test_invalid_inputs_and_serializable_lessons(self):
        for X, y, penalty in [([], [], 0), ([[1]], [1, 2], 0), ([[np.nan]], [1], 0), ([[1]], [1], -1)]:
            with self.assertRaises(ValueError):
                c.ridge_fit(X, y, penalty)
        for number in range(7, 16):
            result = getattr(c, f'lesson_{number:02d}')(42)
            json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
