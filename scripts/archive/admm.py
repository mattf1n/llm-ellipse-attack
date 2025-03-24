import functools as ft, math, time, sys
from tqdm import tqdm
import numpy as np, cvxpy as cp, scipy
import fire

"""
An implementation of the ADMM method for fitting multi-dimensional ellipsoids
as put forward in Lin et al. (2016). Variable names are set according the
method described therein, though hats are skipped.

Z. Lin and Y. Huang, "Fast Multidimensional Ellipsoid-Specific Fitting by
    Alternating Direction Method of Multipliers," in IEEE Transactions on
    Pattern Analysis and Machine Intelligence, vol. 38, no. 5, pp. 1021-1026, 
    1 May 2016, doi: 10.1109/TPAMI.2015.2469283.
"""


def get_G_inv(U, Lambda, beta):
    return U @ np.linalg.inv(2 * np.diag(Lambda) + beta * np.eye(len(Lambda))) @ U.T


def get_G_inv_times(U, Lambda, beta, z):
    return U @ (
        np.linalg.inv(2 * np.diag(Lambda) + beta * np.eye(len(Lambda))) @ (U.T @ z)
    )


def update_q(c, s, mu, beta, get_G_inv, get_G_inv_times):
    g = mu + beta * s
    G_inv = get_G_inv(beta)
    G_inv_g = get_G_inv_times(beta, g)
    G_inv_c = get_G_inv_times(beta, c)
    return G_inv @ ((1 - c.T @ G_inv_g) / (c.T @ G_inv_c) + g)


def main(p: int = 50):
    m = (p + 2) * (p + 1) // 2
    n = m
    X = np.random.random((n, p))
    c = np.array([i == j and i < p for i in range(p + 1) for j in range(i + 1)])
    print(f"Constructing D with shape {(n, m)}", file=sys.stderr)
    quadratic_terms = (
        X[:, None, :]
        * X[:, :, None]
        * (np.eye(p) + np.triu(np.sqrt(2) * np.ones((p, p)), 1))
    )
    quadratic_terms = quadratic_terms[:, *np.triu_indices(p)]
    D = np.block([quadratic_terms, X, np.ones((n, 1))])
    assert D.shape == (n, m), f"{D.shape=}. It should be {(n,m)}."
    print("Constructing K", file=sys.stderr)
    start = time.time()
    K = D.T @ D
    print(f"Took {time.time() - start} seconds", file=sys.stderr)
    assert np.allclose(K, K.T), f"K should be symmetric. {K=}"
    epsilon1 = epsilon2 = 1e-3
    beta_k = 0.1
    rho_k = 1.02
    mu_k = s_k = q_k = np.zeros(m)
    print(f"Getting eigendecomposition of K", file=sys.stderr)
    eigh_start = time.time()
    Lambda, U = np.linalg.eigh(K)
    print(f"{p} Took {time.time() - eigh_start} seconds", file=sys.stderr)
    assert np.allclose(U @ np.diag(Lambda) @ U.T, K)
    print(f"{D.shape=}, {K.shape=}, {Lambda.shape=}, {U.shape=}, {mu_k.shape=}", file=sys.stderr)
    done = True
    while not done:
        q_kp1 = update_q(
            c,
            s_k,
            mu_k,
            beta_k,
            ft.partial(get_G_inv, U, Lambda),
            ft.partial(get_G_inv_times, U, Lambda),
        )
        s_kp1 = update_s()
        mu_kp1 = update_mu()
        beta_kp1 = update_beta()
        done = (
            np.linalg.norm(s_kp1 - q_kp1, np.inf) < epsilon1
            and max(np.linalg.norm(q_kp1 - q_k, np.inf), inf_norm(s_kp1 - s_k, np.inf))
            < epsilon2
        )
        q_k = q_kp1
        s_k = s_kp1
        mu_k = mu_kp1
        beta_k = beta_kp1
    runtime = time.time() - start
    return runtime


if __name__ == "__main__":
    # fire.Fire(main)
    for p in [8, 16, 32, 48, 64, 80, 96, 112, 128]:
        print(p, main(p), flush=True)
