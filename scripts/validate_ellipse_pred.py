
import os, sys, re
from glob import glob
import numpy as np
import pandas as pd
from ellipse_attack.transformations import Model, Ellipse

narrow_band = "--filter" in sys.argv

model_params = np.load("data/model/TinyStories-1M.npz")
model = Model(**model_params)
true_ellipse = model.ellipse()

ellipses = []
# samples_list = [5000, 10_000, 20_000, 30_000, None]
ellipse_pred_filenames = (
    glob("data/narrow_band_ellipse_pred_*_samples.npz")
    if narrow_band
    else glob("data/ellipse_pred_*_samples.npz")
)
for filename in ellipse_pred_filenames:
    # Load ellipse predictions
    params = np.load(filename)["arr_0"]
    ellipse = Ellipse(**params)
    ellipses.append(ellipse)

sample_size_extractor = (
    lambda x: re.compile("data/(narrow_band_)?ellipse_pred_(.*)_samples.npz")
    .match(x)
    .group(2)
)
data = {
    ("Samples", None): [
        sample_size_extractor(fname) for fname in ellipse_pred_filenames
    ],
    ("RMS", "rot"): [
        np.sqrt(np.mean(np.square(ellipse.rot2 - true_ellipse.rot2)))
        for ellipse in ellipses
    ],
    ("RMS", "stretch"): [
        np.sqrt(np.mean(np.square(ellipse.stretch - true_ellipse.stretch)))
        for ellipse in ellipses
    ],
    ("RMS", "bias"): [
        np.sqrt(np.mean(np.square(ellipse.bias - true_ellipse.bias)))
        for ellipse in ellipses
    ],
    ("Max rel. diff.", "rot"): [
        np.max(np.abs(ellipse.rot2 - true_ellipse.rot2))
        for ellipse in ellipses
    ],
    ("Max rel. diff.", "S"): [
        np.max(np.abs(ellipse.stretch - true_ellipse.stretch))
        for ellipse in ellipses
    ],
    ("Max rel. diff.", "bias"): [
        np.max(np.abs(ellipse.bias - true_ellipse.bias))
        for ellipse in ellipses
    ],
}
output_filename = (
    "overleaf/tab/narrow_band_errors.tex"
    if narrow_band
    else "overleaf/tab/errors.tex"
)
for key, value in data.items():
    print(key, len(value))
pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(data.keys())).to_pickle(
    "data/narrow_band_error_data.pkl" if narrow_band else "data/error_data.pkl"
)
# .format(precision=4).to_latex(output_filename)
