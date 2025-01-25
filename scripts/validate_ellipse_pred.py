import numpy as np
import pandas as pd
from get_ellipse import get_transform

model_params = np.load("data/model_params.npz")
W = model_params["W"]
gamma = model_params["gamma"]
beta = model_params["beta"]
logits = model_params["logits"]
hidden = model_params["hidden"]
prenorm = model_params["prenorm"]

d, v = W.shape
n = d - 1

U_preds, S_preds, bias_preds = [], [], []
samples_list = [5000, 10_000, 20_000, 30_000, None]
for samples in samples_list:
    ellipse_preds = np.load(f"data/ellipse_pred_{samples}_samples.npz")
    S_pred = ellipse_preds["S_pred"]
    U_pred = ellipse_preds["U_pred"]
    bias_pred = ellipse_preds["bias_pred"]
    S_preds.append(S_pred)
    U_preds.append(U_pred)
    bias_preds.append(bias_pred)
    inverted = (
        (logits[:, :n] - bias_preds)
        @ np.linalg.inv(U_pred)
        @ np.linalg.inv(np.diag(S_pred))
    )
    true = ((hidden - beta) @ np.linalg.inv(np.diag(gamma)))[:, :n]
    true = true / np.linalg.norm(true, axis=1, keepdims=True)
    soln, *_ = np.linalg.lstsq(inverted, true)
    np.testing.assert_allclose(soln.T @ soln, np.eye(n))

    bias = beta @ W[:, :n]
    project_to_sphere = get_transform(np.ones(d), np.arange(d) == n)
    linear = np.linalg.inv(project_to_sphere)[:n, :] @ np.diag(gamma) @ W[:, :n]
    Vh, S, U_ = np.linalg.svd(linear)
    U = np.diag((U_[:, 0] > 0) * 2 - 1) @ U_
    C = np.linalg.inv(linear.T @ linear)
    unbiased = logits[:, :n] - bias
    sphere = (prenorm - prenorm.mean(axis=1, keepdims=True)) / np.sqrt(
        prenorm.var(axis=1, keepdims=True, ddof=1) + 1e-5
    )
    testing = False
    if testing:
        np.testing.assert_allclose(
            np.linalg.norm(sphere, axis=1),
            np.sqrt(d - 1),
            atol=1e-1,
            err_msg="sphere not sphereing",
        )
        np.testing.assert_allclose(
            np.linalg.norm((hidden - beta) / gamma, axis=1),
            np.sqrt(d - 1),
            atol=1e-1,
            err_msg="hidden sphere not sphereing",
        )
        np.testing.assert_allclose(
            logits[:, :n],
            hidden @ W[:, :n],
            err_msg="params don't work",
        )
        np.testing.assert_allclose(
            np.einsum("ij,ij->i", unbiased @ C, unbiased),
            1,
            atol=1e-1,
            err_msg="Inversion wrong",
        )

data = {
    ("Samples", None): samples_list,
    ("Mean RMS", "U"): [np.sqrt(np.mean(np.square(U - U_pred))) for U_pred in U_preds],
    ("Mean RMS", "S"): [np.sqrt(np.mean(np.square(S - S_pred))) for S_pred in S_preds],
    ("Mean RMS", "bias"): [
        np.sqrt(np.mean(np.square(bias - bias_pred))) for bias_pred in bias_preds
    ],
    ("Max relative difference", "U"): [
        np.max(np.abs(U - U_pred) / U) for U_pred in U_preds
    ],
    ("Max relative difference", "S"): [np.max((S - S_pred) / S) for S_pred in S_preds],
    ("Max relative difference", "bias"): [
        np.max(np.abs(bias - bias_pred) / bias) for bias_pred in bias_preds
    ],
}
pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(data.keys())).style.hide(
    axis="index"
).to_latex("tab/errors.tex")
