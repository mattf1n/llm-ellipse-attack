import math
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from get_ellipse import get_ellipse, get_transform

d: int = 20
v: int = d * 100
n: int = d - 1
m: int = (n + 1) * (n + 2) // 2
k: int = (math.comb(n + 1, 2) + n) * 10
r = c = math.ceil(math.sqrt(k))

inputs = np.random.normal(size=(k, d))
standardized = (inputs - inputs.mean(1, keepdims=True)) / np.sqrt(
    inputs.var(1, keepdims=True, ddof=1) + 1e-5
)

W = np.random.normal(size=(d, v))
gamma = np.random.normal(size=d)
beta = np.random.normal(size=d)
logits = (standardized @ np.diag(gamma) + beta) @ W  # k, v
x = logits[:, :d]

project_to_sphere = get_transform(np.ones(d), np.arange(d) == n)
sphere = standardized @ project_to_sphere[:, :n]
bias = (beta @ W)[:n]
ellipse_to_image, resids, *_ = np.linalg.lstsq(
    np.hstack([x[:, :n], np.ones((k, 1))]), x
)
image_to_vocab, resids, *_ = np.linalg.lstsq(x, logits)  # d, v
linear = np.linalg.inv(project_to_sphere)[:n, :] @ np.diag(gamma) @ W[:, :n]  # n, d
C = np.linalg.inv(linear.T @ linear)

logits_ = logits[:, :n]
for k_subset in range(1):
    C_pred, S_pred, U_pred, bias_pred = get_ellipse(logits_)
    idxs = np.argsort(
        -np.linalg.norm(
            (logits[:, :n] - bias_pred)
            @ np.linalg.inv(U_pred)
            @ np.linalg.inv(np.diag(S_pred)),
            axis=1,
        )
    )
    logits_ = logits[idxs[: k - (k * k_subset // 100)], :n]


Vh_, S, U_ = np.linalg.svd(linear)
reflection = np.diag((U_[:, 0] > 0) * 2 - 1)
Vh, U = Vh_ @ reflection, reflection @ U_

plt.hist(S_pred - S)
plt.show()

plt.hist(
    np.einsum(
        "ij,ij->i", (logits[:, :n] - bias_pred) @ C_pred, logits[:, :n] - bias_pred
    ),
)
plt.ticklabel_format(useOffset=False)
plt.show()

np.testing.assert_allclose(
    S_pred * 1.0545,
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


print("All tests passed!")
