import numpy as np
X = np.array([[2., 3.], [0., 1.], [4., 2.]])
w = np.array([4., -1.])
b = 7.0
prediction = X @ w + b
assert prediction.shape == (3,)
assert np.allclose(prediction, [12., 6., 21.])
