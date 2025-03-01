
import os, sys, re
from glob import glob
import numpy as np
import pandas as pd
import fire
from ellipse_attack.transformations import Model, Ellipse



def main(*fnames, narrow_band: bool = False):
    model_params = np.load("data/model/TinyStories-1M.npz")
    model = Model(**model_params)

    ellipses = []
    for filename in fnames:
        # Load ellipse predictions
        params = np.load(filename, allow_pickle=True)
        down_proj = params["down_proj"]
        ellipse = Ellipse(
                up_proj=params["up_proj"],
                bias=params["bias"],
                rot1=params["rot1"],
                stretch=params["stretch"],
                rot2=params["rot2"],
                )
        true_ellipse = model.ellipse(down_proj=down_proj if None not in down_proj else None)
        ellipses.append((ellipse, true_ellipse))

    sample_size_extractor = (
        lambda x: re.compile("data/(.*)?ellipse_pred_(.*)_samples(.*).npz")
        .match(x)
        .group(2)
    )
    param_names = ("stretch", "rot2", "bias")
    data = {
            ("fnames", None): list(map(os.path.basename, fnames)),
            ("Samples", None): [
                sample_size_extractor(fname) for fname in fnames
                ],
            ("Angle", "rot2"): [
                np.rad2deg(np.arccos((np.trace(ellipse.rot2.T @ true_ellipse.rot2) - 1) / 2))
                for ellipse, true_ellipse in ellipses
                ],
            **{
                ("RMS", param): [
                    np.sqrt(np.mean(np.square(getattr(ellipse, param) - getattr(true_ellipse, param))))
                    for ellipse, true_ellipse in ellipses
                    ]
                for param in param_names
                },
            **{
                ("Max diff.", param): [
                    np.max(np.abs(getattr(ellipse, param) - getattr(true_ellipse, param)))
                    for ellipse, true_ellipse in ellipses
                    ]
                for param in param_names
                },
            **{
                ("Mean diff.", param): [
                    np.mean(np.abs(getattr(ellipse, param) - getattr(true_ellipse, param)))
                    for ellipse, true_ellipse in ellipses
                    ]
                for param in ("stretch", "rot2", "bias")
                },
            **{
                ("Mean rel. diff.", param): [
                    np.mean(
                        np.abs(
                            (getattr(ellipse, param) - getattr(true_ellipse, param))
                            / getattr(true_ellipse, param)
                            ) 
                        )
                    for ellipse, true_ellipse in ellipses
                    ]
                for param in param_names
                },
            **{
                ("Max rel. diff.", param): [
                    np.max(
                        np.abs(
                            (getattr(ellipse, param) - getattr(true_ellipse, param))
                            / getattr(true_ellipse, param)
                            ) 
                        )
                    for ellipse, true_ellipse in ellipses
                    ]
                for param in param_names
                },
            }
    output_filename = (
        "overleaf/tab/narrow_band_errors.tex"
        if narrow_band
        else "overleaf/tab/errors.tex"
    )
    df = pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(data.keys()))
    print(df.T)
    df.to_pickle(
        "data/narrow_band_error_data.pkl" if narrow_band else "data/error_data.pkl"
    )
    # .format(precision=4).to_latex(output_filename)

if __name__ == "__main__":
    fire.Fire(main)
