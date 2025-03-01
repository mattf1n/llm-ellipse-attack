import numpy as np

import ellipse_attack.metrics as metrics


def test_angle():
    n = 30
    matrix = np.arange(pow(n, 2)).reshape(n, n) - n / 2
    U, S, Vh = np.linalg.svd(matrix)
    np.testing.assert_allclose(metrics.angle(U, U) == 0)
