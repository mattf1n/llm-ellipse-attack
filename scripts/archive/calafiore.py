import math
import numpy as np
import cvxpy as cp

dim: int = 10
sample_size: int = 2 * (math.comb(dim, 2) + dim)
gamma = cp.Variable(sample_size, pos=True)
A = cp.Variable((dim, dim), PSD=True)
b = cp.Variable(dim)
d = cp.Variable((1,))
print(A.shape, b[None, :].shape)
omega = cp.bmat([[A, b[:, None]], [b[None, :], d[None, :]]])

inputs = np.random.random((sample_size, dim + 1))
standardized = (
    (inputs - inputs.mean(1, keepdims=True))
    / inputs.std(1, keepdims=True)
    @ np.eye(dim + 1, dim)
)
W = np.random.random((dim, dim))
beta = np.random.random(dim)
C = np.linalg.inv(W @ W.T)
U, S, Vh = np.linalg.svd(C)
x = standardized @ W + beta


def g(x):
    x1 = np.block([x, 1])
    return x1[None, :] @ omega @ x1


def construct(i):
    result = cp.bmat([[1, g(x[i])], [g(x[i]), gamma[i]]])
    return result


psd_constraints = [construct(i) >> 0 for i in range(sample_size)]
other_constraints = [cp.trace(A) == 1]
objective = cp.Minimize(cp.sum(gamma))
problem = cp.Problem(objective, psd_constraints + other_constraints)
problem.solve(solver="MOSEK", verbose=True, canon_backend="CPP")

c = -np.linalg.inv(A.value) @ b.value
r_squared = c @ A.value @ c - d.value
r_squared_pred = np.vecdot(x - c, (x - c) @ A.value)
assert np.allclose(r_squared_pred, r_squared, rtol=1e-4)
print(r_squared_pred)
print(r_squared)

print(c)
print(beta)
_, S_, _ = np.linalg.svd(A.value)
print(S / np.linalg.norm(S))
print(S_ / np.linalg.norm(S_))
