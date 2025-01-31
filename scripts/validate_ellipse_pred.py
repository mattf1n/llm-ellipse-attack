"""
# Standard formulation

```
prenorm (hidden_size) 
-standardize-> standardized (hidden_size)
-gamma+beta-> hidden (hidden_size)
-W-> logit (vocab_size)
```
where `standardize` is
```
prenorm -center-> centered -normalize*sqrt(hidden_size)-> standardized
```

# Alt. formulation

```
prenorm (hidden_size) 
-standardize+normalize+Vh-> sphere_projection (ellipse_rank)
-S-> stretched (ellipse_rank)
-U-> rotated (vocab or ellipse_rank)
-bias-> logit (vocab or ellipse_rank)
```

`vocab or ellipse_rank` sizes are due to the fact that we are not finding the ellipse for the whole vocabulary, just `ellipse_rank` of them.


"""

import os
import numpy as np
import pandas as pd
from get_ellipse import get_transform

narrow_band = True

model_params = np.load("data/vocab/model_params.npz")
W = model_params["W"]
gamma = model_params["gamma"]
beta = model_params["beta"]
logits = model_params["logits"]
hidden = model_params["hidden"]
prenorm = model_params["prenorm"]

data_size = logits.shape[0]
hidden_size, v = W.shape
ellipse_rank = hidden_size - 1

U_preds, S_preds, bias_preds = [], [], []
samples_list = [5000, 10_000, 20_000, 30_000, None]
for sample_size in samples_list:
    # Load ellipse predictions
    ellipse_pred_file = (
        f"data/narrow_band_ellipse_pred_{sample_size}_samples.npz"
        if narrow_band
        else f"data/ellipse_pred_{sample_size}_samples.npz"
    )
    if not os.path.exists(ellipse_pred_file):
        continue
    ellipse_preds = np.load(ellipse_pred_file)
    S_pred = ellipse_preds["S_pred"]
    U_pred = ellipse_preds["U_pred"]
    bias_pred = ellipse_preds["bias_pred"]
    print(f"{S_pred.shape=}, {U_pred.shape=}, {bias_pred.shape=}")
    S_preds.append(S_pred)
    U_preds.append(U_pred)
    bias_preds.append(bias_pred)

    print(f"{logits.shape=}")
    sphere_projection_pred = (
        (logits[:, :ellipse_rank] - bias_pred)
        @ np.linalg.inv(U_pred)
        @ np.linalg.inv(np.diag(S_pred))
    )
    np.testing.assert_allclose(
        np.linalg.norm(sphere_projection_pred, axis=1),
        1.0,
        atol=6e-2,
        err_msg="Sphere projection predictions are not on the sphere.",
    )
    standardized = ((hidden - beta) @ np.linalg.inv(np.diag(gamma))) / np.sqrt(hidden_size)
    standardized_down_proj = ... # TODO use get_transform to project this down along the ones vec.
    np.testing.assert_allclose(
        np.linalg.norm(standardized, axis=1),
        1.0,
        atol=2e-2,
        err_msg="Standardized values are not on the sphere.",
    )
    print(f"{standardized.shape=}")
    soln, *_ = np.linalg.lstsq(sphere_projection_pred, standardized)
    # TODO Revisit this non-passing test case
    np.testing.assert_allclose(soln.T @ soln, np.eye(ellipse_rank+1))

    bias = beta @ W[:, :ellipse_rank]
    project_to_sphere = get_transform(
        np.ones(hidden_size), np.arange(hidden_size) == ellipse_rank
    )
    linear = (
        np.linalg.inv(project_to_sphere)[:ellipse_rank, :]
        @ np.diag(gamma)
        @ W[:, :ellipse_rank]
    )
    Vh, S, U_ = np.linalg.svd(linear)
    U = np.diag((U_[:, 0] > 0) * 2 - 1) @ U_
    C = np.linalg.inv(linear.T @ linear)
    unbiased = logits[:, :ellipse_rank] - bias
    sphere = (prenorm - prenorm.mean(axis=1, keepdims=True)) / np.sqrt(
        prenorm.var(axis=1, keepdims=True, ddof=1) + 1e-5
    )
    # TODO add test case here to compare standardized and true sphere.
    testing = False
    if testing:
        np.testing.assert_allclose(
            np.linalg.norm(sphere, axis=1),
            np.sqrt(hidden_size - 1),
            atol=1e-1,
            err_msg="sphere not sphereing",
        )
        np.testing.assert_allclose(
            np.linalg.norm((hidden - beta) / gamma, axis=1),
            np.sqrt(hidden_size - 1),
            atol=1e-1,
            err_msg="hidden sphere not sphereing",
        )
        np.testing.assert_allclose(
            logits[:, :ellipse_rank],
            hidden @ W[:, :ellipse_rank],
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
output_filename = (
    "overleaf/tab/narrow_band_errors.tex" if narrow_band else "overleaf/tab/errors.tex"
)
pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(data.keys())).style.hide(
    axis="index"
).to_latex(output_filename)
