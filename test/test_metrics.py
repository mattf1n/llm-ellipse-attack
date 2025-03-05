import numpy as np

import ellipse_attack.metrics as metrics


def test_angle():
    n = 11
    matrix1 = np.arange(pow(n, 2)).reshape(n, n)
    matrix2 = np.arange(pow(n, 2)).reshape(n, n) - 0.5
    matrix3 = np.arange(pow(n, 2)).reshape(n, n) - pow(n, 2) / 2
    U1, S, Vh1 = np.linalg.svd(matrix1)
    U2, S, Vh2 = np.linalg.svd(matrix2)
    U3, S, Vh3 = np.linalg.svd(matrix3)
    np.testing.assert_allclose(metrics.angle(U1, U1), 0, atol=1e-5)
    assert metrics.angle(U1, U2) > 0
    assert metrics.angle(U1, np.eye(n)) == np.sqrt(
            np.square(np.maximum(0, np.angle(np.linalg.eigvals(U1)))).sum()
            )
    vector = np.ones(n)
    assert (U1 @ vector) @ (U1 @ vector) > (U1 @ vector) @ (U2 @ vector)
    assert metrics.angle(U1, np.eye(n)) < metrics.angle(U3, np.eye(n))
