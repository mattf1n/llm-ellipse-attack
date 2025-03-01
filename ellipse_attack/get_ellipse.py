import math
import numpy as np
import cvxpy as cp
from jaxtyping import Num, Array


def get_ellipse(x, **kwargs):
    """
    Takes a set of k points on an n-dimensions sphere and computes
    C: ??
    U: the rotation
    S: the singular values
    bias: the bias
    """
    k, n = x.shape
    r = c = math.ceil(math.sqrt(k))

    A, b, dee = fit_ellipse(x, n, r, c, **kwargs)
    bias = -np.linalg.inv(A) @ b
    r_squared = np.abs(bias @ A @ bias - dee)
    C = A / r_squared
    Cinv = np.linalg.inv(C)

    try:
        linear = np.linalg.cholesky(Cinv).T
    except np.linalg.LinAlgError as e:
        print(C)
        print(Cinv)
        print(Cinv.min())
        raise e
    Vh, S, U_ = np.linalg.svd(linear)
    U = np.diag((U_[:, 0] > 0) * 2 - 1) @ U_
    return C, S, U, bias


def Arc(resid, r, c):
    """Packs the residuals into an $r\\times c$ matrix by padding with zeros"""
    (k,) = resid.shape
    if r * c == k:
        block = resid
    else:
        block = cp.bmat([[resid, np.zeros(r * c - k)]])
    return cp.reshape(block, (r, c), order="C")


def residuals(x, Q):
    k, _ = x.shape
    x_bias = cp.hstack([x, np.ones((k, 1))])
    return cp.sum(cp.multiply(x_bias @ Q, x_bias), axis=1)


def fit_ellipse(x, n, r, c, **kwargs):
    """
    Takes $k\\times d$ data matrix containing points on an ellipsoid
    and returns A, b, d such that
    ```
    Q == [[A   b]
          [b.T d]]
    x.T @ Q @ x == 0
    ```
    """
    t = cp.Variable()
    A = cp.Variable((n, n), PSD=True)
    b = cp.Variable(n)
    d = cp.Variable()
    Q = cp.bmat([[A, b[:, None]], [b[None, :], d[None, None]]])
    constraints = [
        cp.bmat(
            [
                [t * np.eye(c), Arc(residuals(x, Q), r, c).T],
                [Arc(residuals(x, Q), r, c), t * np.eye(r)],
            ]
        )
        >> 0,
        cp.trace(A) == 1,
    ]
    objective = cp.Minimize(t)
    problem = cp.Problem(objective, constraints)
    kwargs = dict(solver="MOSEK", canon_backend="CPP", verbose=True) | kwargs
    problem.solve(**kwargs)
    return A.value, b.value, d.value
