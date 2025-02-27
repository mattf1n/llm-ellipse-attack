import sys
import numpy as np
import ellipse_attack.transformations as tfm


def assert_rotation(a, b):
    soln = np.linalg.solve(a, b)
    np.testing.assert_allclose(soln.T @ soln, np.eye(soln.shape))
    np.testing.assert_allclose(np.linalg.det(soln), 1.0)


def test_lstsq_pinv_equiv():
    A = rng.standard_normal((5, 4))
    B = rng.standard_normal((5, 5))
    C, *_ = np.linalg.lstsq(A, B, rcond=None)
    D = np.linalg.pinv(A) @ B
    np.testing.assert_allclose(C, D)


def test_sphere_point():
    sphere_point = tfm.standardize(test_point)
    assert sphere_point.shape == (emb_size,)
    np.testing.assert_allclose(np.linalg.norm(sphere_point), np.sqrt(emb_size))


def test_sphere_data():
    stnd_test_data = tfm.standardize(test_data)
    assert stnd_test_data.shape[-1] == emb_size
    np.testing.assert_allclose(
        np.linalg.norm(stnd_test_data, axis=-1), np.sqrt(emb_size)
    )


def test_ellipse_of_model():
    ellipse = model.ellipse()
    np.testing.assert_allclose(ellipse(sphere_point), model(sphere_point))


def test_ellipse_of_model_linear_term():
    ellipse = model.ellipse()
    model_side = (
        tfm.isom_inv(emb_size)
        @ np.diag(model.stretch)
        @ model.unembed
        @ tfm.center(vocab_size)
        @ tfm.ctr_to_alr(vocab_size)
        @ np.linalg.pinv(ellipse.up_proj)
    )
    ellipse_side = ellipse.rot1 @ np.diag(ellipse.stretch) @ ellipse.rot2
    np.testing.assert_allclose(model_side, ellipse_side)


def test_ellipse_of_model_linear_term_test_point():
    model_side = (
        sphere_point @ np.diag(model.stretch) @ model.unembed @ tfm.center(vocab_size)
        @ tfm.ctr_to_alr(vocab_size)
    )
    ellipse_side = (
        sphere_point
        @ isom
        @ ellipse.rot1
        @ np.diag(ellipse.stretch)
        @ ellipse.rot2
        @ ellipse.up_proj
    )
    np.testing.assert_allclose(model_side, ellipse_side, rtol=1e-10)


def test_ellipse_up_proj_pinv():
    ellipse = model.ellipse()
    np.testing.assert_allclose(
        ellipse.up_proj @ np.linalg.pinv(ellipse.up_proj),
        np.eye(emb_size - 1),
        atol=1e-10,
    )


def test_isometric_transform_inverse():
    isom_inv = tfm.isom_inv(emb_size)
    centered = tfm.center(emb_size)
    np.testing.assert_allclose(centered @ isom @ isom_inv, centered, atol=1e-10)


def test_isometric_transform_mean_invariance():
    zeros = tfm.isometric_transform(np.ones(10))
    np.testing.assert_allclose(zeros, np.zeros(9), atol=1e-10)
    vector = np.arange(100) - 50
    np.testing.assert_allclose(
        tfm.isometric_transform(vector),
        tfm.isometric_transform(vector - vector.mean()),
        atol=1e-10,
    )


def test_centering_matrix():
    n = 100
    vector = np.arange(n)
    center = tfm.center(n)
    np.testing.assert_allclose(
        vector - vector.mean(), vector @ center, atol=1e-10
    )


def test_ellipse_from_data():
    test_logprobs = model(stnd_test_data)
    from_data = tfm.Ellipse.from_data(
        test_logprobs, emb_size=emb_size, verbose=False
    )
    from_model = model.ellipse()
    np.testing.assert_allclose(from_data.bias, from_model.bias)
    np.testing.assert_allclose(from_data.up_proj, from_model.up_proj, atol=1e-10)
    np.testing.assert_allclose(from_data.stretch, from_model.stretch)
    np.testing.assert_allclose(from_data.rot2, from_model.rot2)


emb_size = 10
vocab_size = 100
scale = 10
sample_size = pow(emb_size, 2)
isom = tfm.isom(emb_size)
rng = np.random.default_rng(10)
model = tfm.Model(
    stretch=rng.normal(size=emb_size, scale=scale),
    bias=rng.normal(size=emb_size, scale=scale),
    unembed=rng.normal(size=(emb_size, vocab_size), scale=scale),
)
ellipse = model.ellipse()
test_point = rng.normal(size=emb_size, scale=scale)
sphere_point = tfm.standardize(test_point)

test_data = rng.normal(size=(sample_size, emb_size), scale=scale)
stnd_test_data = tfm.standardize(test_data)
