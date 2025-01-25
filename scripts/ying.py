import math
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from get_ellipse import get_ellipse

d: int = 200
v: int = d * 100
n: int = d - 1
m: int = (n + 1) * (n + 2) // 2
k: int = (math.comb(n + 1, 2) + n) * 10
r = c = math.ceil(math.sqrt(k))


def reflect(A, n):
    return A - 2 * np.outer(n, (n @ A) / (n @ n))


def get_transform(u, v):
    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)
    u = u / u_norm
    v = v / v_norm
    S = reflect(np.eye(len(u)), u + v)
    R = reflect(S, v) * v_norm / u_norm
    return R.T


def Arc(resid):
    resid  # k,
    if r * c == k:
        padded = resid
    else:
        padded = cp.bmat([[resid, np.zeros(r * c - k)]])
    return cp.reshape(padded, (r, c))


def residuals(x, Q):  # k
    x_ = cp.hstack([x, np.ones((k, 1))])
    return cp.sum(cp.multiply(x_ @ Q, x_), axis=1)


def fit_ellipse(x):
    t = cp.Variable()
    A = cp.Variable((n, n), PSD=True)
    b = cp.Variable(n)
    d = cp.Variable()
    Q = cp.bmat([[A, b[:, None]], [b[None, :], d[None, None]]])
    constraints = [
        cp.bmat(
            [
                [t * np.eye(c), Arc(residuals(x, Q)).T],
                [Arc(residuals(x, Q)), t * np.eye(r)],
            ]
        )
        >> 0,
        cp.trace(A) == 1,
    ]
    objective = cp.Minimize(t)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver="MOSEK", canon_backend="CPP", verbose=True)
    return A.value, b.value, d.value


inputs = np.random.random((k, d))
standardized = (inputs - inputs.mean(1, keepdims=True)) / np.sqrt(
    inputs.var(1, keepdims=True, ddof=1) + 1e-5
)
np.testing.assert_allclose(np.einsum("ij,ij->i", standardized, standardized), d - 1)

W = np.random.random((d, v)) * 2
gamma = np.random.random(d) * 2
beta = np.random.random(d) * 2
logits = (standardized @ np.diag(gamma) + beta) @ W  # k, v
x = logits[:, :d]

project_to_sphere = get_transform(np.ones(d), np.arange(d) == n)
sphere = standardized @ project_to_sphere[:, :n]
np.testing.assert_allclose(
    np.ones(d) @ project_to_sphere[:, :n],
    np.zeros(n),
    atol=1e-10,
    err_msg="Project to sphere broken",
)
np.testing.assert_allclose(
    np.linalg.norm(sphere, axis=1),
    1,
    err_msg="Project to sphere broken",
)
np.testing.assert_allclose(
    sphere @ np.linalg.inv(project_to_sphere)[:n, :],
    standardized,
    err_msg="Project to sphere inv broken",
)
np.testing.assert_allclose(
    np.linalg.inv(project_to_sphere)[:n, :] @ project_to_sphere[:, :n],
    np.eye(n),
    atol=1e-10,
    err_msg="Project to sphere inv broken",
)


bias = (beta @ W)[:n]

ellipse_to_image, resids, *_ = np.linalg.lstsq(
    np.hstack([x[:, :n], np.ones((k, 1))]), x
)
np.testing.assert_allclose(resids, 0, atol=1e-10)

image_to_vocab, resids, *_ = np.linalg.lstsq(x, logits)  # d, v
np.testing.assert_allclose(resids, 0, atol=1e-10)

linear = np.linalg.inv(project_to_sphere)[:n, :] @ np.diag(gamma) @ W[:, :n]  # n, d
np.testing.assert_allclose(
    logits,
    np.hstack([sphere @ linear + bias, np.ones((k, 1))])
    @ ellipse_to_image
    @ image_to_vocab,
    atol=1e-6,
    err_msg="Reparameterization incorrect",
)

C = np.linalg.inv(linear.T @ linear)
np.testing.assert_allclose(
    np.einsum("ij,ij->i", (logits[:, :n] - bias) @ C, logits[:, :n] - bias),
    1,
    atol=1e-6,
    err_msg="True C is not valid",
)

A, b, dee = fit_ellipse(logits[:, :n])
bias_pred = -np.linalg.inv(A) @ b
r_squared = np.abs(bias_pred @ A @ bias_pred - dee)
C_pred = A / r_squared

np.testing.assert_allclose(
    np.einsum("ij,ij->i", (logits[:, :n] - bias_pred) @ A, logits[:, :n] - bias_pred),
    r_squared,
    atol=1e-3,
    err_msg="C_pred is not correct w.r.t. r^2",
)

np.testing.assert_allclose(
    np.einsum(
        "ij,ij->i", (logits[:, :n] - bias_pred) @ C_pred, logits[:, :n] - bias_pred
    ),
    1,
    atol=1e-5,
    err_msg="C_pred is not correct",
)
np.testing.assert_allclose(
    np.einsum(
        "ij,ij->i", (logits[:, :n] - bias_pred) @ C_pred, logits[:, :n] - bias_pred
    ),
    1,
    atol=1e-5,
    err_msg="C_pred is not correct w.r.t. true bias",
)

linear_pred = np.linalg.cholesky(np.linalg.inv(C_pred)).T
Vh_pred, S_pred, U_pred = np.linalg.svd(linear_pred)
reflection_pred = np.diag((U_pred[:, 0] > 0) * 2 - 1)
Vh_pred, U_pred = Vh_pred @ reflection_pred, reflection_pred @ U_pred
np.testing.assert_allclose(
    U_pred[:, 0], np.abs(U_pred[:, 0]), err_msg="Postivity unsuccessful"
)
np.testing.assert_allclose(
    Vh_pred @ np.diag(S_pred) @ U_pred,
    linear_pred,
    err_msg="Reflection unsuccessful",
    atol=1e-10,
)
S_pred, U_pred, bias_pred = get_ellipse(logits[:, :n])
np.testing.assert_allclose(
    U_pred[:, 0], np.abs(U_pred[:, 0]), err_msg="Pred reflection unsuccessful"
)

Vh_, S, U_ = np.linalg.svd(linear)
reflection = np.diag((U_[:, 0] > 0) * 2 - 1)
Vh, U = Vh_ @ reflection, reflection @ U_
np.testing.assert_allclose(
    U[:, 0], np.abs(U[:, 0]), err_msg="True reflection unsuccessful"
)
np.testing.assert_allclose(Vh_ @ reflection @ np.diag(S) @ U, linear)

np.testing.assert_allclose(
    S_pred,
    S,
    atol=1e-4,
    err_msg="Singular value mismatch",
)

np.testing.assert_allclose(
    U_pred,
    U,
    atol=1e-4,
    err_msg="Rotation mismatch",
)


# x_pred = (np.hstack([sphere @ linear + bias, np.ones((k, 1))]) @ ellipse_to_image,)

# np.testing.assert_allclose(
#     x,
#     atol=1e-6,
#     err_msg="Reparameterization incorrect",
# )

# colors = np.arctan2(*sphere.T)
# fig, ax = plt.subplots()
# ax.set_aspect("equal", adjustable="box")
# plt.scatter(*sphere.T, c=colors, cmap="binary")
# plt.scatter(
#     *(sphere @ Vh @ np.diag(S) @ U).T,
#     label="True ellipse",
#     c=colors,
#     cmap="seismic",
#     alpha=0.1
# )
# plt.scatter(
#     *(sphere @ Vh_pred @ np.diag(S_pred) @ U_pred).T,
#     label="Estimated ellipse",
#     c=colors,
#     cmap="afmhot",
#     alpha=0.1
# )
# plt.legend()
# plt.savefig("fig/out.pdf")

print("All tests passed!")
