import math
import numpy as np
import cvxpy as cp


def get_ellipse(x):
    k, n = x.shape
    r = c = math.ceil(math.sqrt(k))

    A, b, dee = fit_ellipse(x, n, r, c)
    bias = -np.linalg.inv(A) @ b
    r_squared = np.abs(bias @ A @ bias - dee)
    C = A / r_squared

    linear = np.linalg.cholesky(np.linalg.inv(C)).T
    Vh, S, U_ = np.linalg.svd(linear)
    U = np.diag((U_[:, 0] > 0) * 2 - 1) @ U_
    return C, S, U, bias


def reflect(A, n):
    """Utility for `get_transform`"""
    return A - 2 * np.outer(n, (n @ A) / (n @ n))


def get_transform(u, v):
    """
    Takes two vectors $u$ and $v$ and returns a matrix that maps $u$ into $v$
    by rotating about the vector $u\\times v$.
    """
    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)
    u = u / u_norm
    v = v / v_norm
    S = reflect(np.eye(len(u)), u + v)
    R = reflect(S, v) * v_norm / u_norm
    return R.T


def isometric_transform(u):
    dim = u.shape[-1]
    transform = get_transform(np.ones(dim), np.arange(dim) == dim)[:, :ellipse_rank]
    # TODO: check the ordering below (and indexing above)
    return u @ transform 


def Arc(resid, r, c):
    """Packs the residuals into an $r\\times c$ matrix by padding with zeros"""
    (k,) = resid.shape
    if r * c == k:
        block = resid
    else:
        block = cp.bmat([[resid, np.zeros(r * c - k)]])
    return cp.reshape(block, (r, c))


def residuals(x, Q):
    k, _ = x.shape
    x_bias = cp.hstack([x, np.ones((k, 1))])
    return cp.sum(cp.multiply(x_bias @ Q, x_bias), axis=1)


def fit_ellipse(x, n, r, c):
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
    problem.solve(solver="MOSEK", canon_backend="CPP", verbose=True)
    return A.value, b.value, d.value
