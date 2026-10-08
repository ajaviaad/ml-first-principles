import unittest
import numpy as np
from mlfirst import foundations as f


class FoundationsTests(unittest.TestCase):
    def test_group_split_and_constant_feature(self):
        groups = np.repeat(np.arange(8), 3)
        a, b = f.split_groups(groups, seed=5)
        self.assertEqual(set(a) | set(b), set(range(24)))
        self.assertFalse(set(groups[a]) & set(groups[b]))
        scaler = f.Standardizer().fit([[1, 7], [3, 7]])
        np.testing.assert_allclose(scaler.transform([[1, 7], [3, 7]]), [[-1, 0], [1, 0]])
        mean = scaler.mean_.copy()
        scaler.transform([[1000, 7]])
        np.testing.assert_array_equal(scaler.mean_, mean)

    def test_shapes_do_not_broadcast(self):
        with self.assertRaises(ValueError):
            f.regression_metrics([1, 2], [[1], [2]])
        with self.assertRaises(ValueError):
            f.Standardizer().transform([[1, 2]])

    def test_missing_group_ids_cannot_silently_empty_holdout(self):
        for groups in ([1, np.nan, np.nan], [1, np.inf], ["a", None],
                       [np.datetime64("2020-01-01"), np.datetime64("NaT")]):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                f.split_groups(groups)
        train, test = f.split_groups(["a", "a", "b", "b"])
        self.assertEqual(len(train), 2)
        self.assertEqual(len(test), 2)

    def test_derivatives_and_stability(self):
        grad = f.central_difference(lambda x: 3*x[0]**2 + x[0]*x[1], [2, 4])
        np.testing.assert_allclose(grad, [16, 2], atol=1e-8)
        self.assertAlmostEqual(f.logsumexp([1000, 1000]), 1000 + np.log(2))
        np.testing.assert_allclose(f.quadratic_descent(2, .25, 3, 3), [3, 1.5, .75, .375])

    def test_bayes_and_bootstrap(self):
        self.assertAlmostEqual(f.bayes_positive(.01, .9, .05), 90/585)
        self.assertEqual(f.bootstrap_mean([3, 3, 3]), [3, 3])
        with self.assertRaises(ValueError):
            f.bayes_positive(0, 0, 0)

    def test_metrics_against_hand_counts(self):
        m = f.binary_metrics([0, 0, 1, 1], [.2, .7, .4, .8])
        self.assertEqual([m[k] for k in ("tp", "fp", "fn", "tn")], [1, 1, 1, 1])
        self.assertEqual(m["roc_auc"], .75)
        tied = f.binary_metrics([0, 1], [.5, .5])
        self.assertEqual(tied["roc_auc"], .5)
        self.assertIsNone(f.binary_metrics([0, 0], [0, 0])["recall"])
        self.assertIsNone(f.regression_metrics([1, 1], [2, 2])["r2"])
        self.assertLess(f.regression_metrics([1, 2], [10, 10])["r2"], 0)

    def test_conformal_rank(self):
        self.assertEqual(f.conformal_radius(range(1, 20)), 18)
        self.assertTrue(np.isinf(f.conformal_radius([1, 2], alpha=.1)))
        with self.assertRaises(ValueError):
            f.conformal_radius([-1, 2])


if __name__ == "__main__":
    unittest.main()
