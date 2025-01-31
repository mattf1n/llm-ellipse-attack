import numpy as np
import ellipse_attack.get_ellipse as ea


def test_isometric_transform():
    zeros = ea.isometric_transform(np.ones(10))
    np.testing.assert_allclose(zeros, np.zeros(9), atol=1e-10)
    vector = np.arange(100) - 50
    np.testing.assert_allclose(
            ea.isometric_transform(vector),
            ea.isometric_transform(vector - vector.mean()),
            atol=1e-10
            )
