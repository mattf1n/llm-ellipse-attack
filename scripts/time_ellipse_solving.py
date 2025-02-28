import itertools as it, sys, time
from dataclasses import asdict
import numpy as np
import scipy
from ellipse_attack.transformations import Ellipse
import fire


def main(sample_size: int, narrow_band: bool = False):
    hidden_size = 64
    fname = (
        "data/single_token_prompts/narrow_band_logprobs.npy"
        if narrow_band
        else "data/single_token_prompts/logprobs.npy"
    )
    logprobs = np.load(fname)

    ellipse_rank = hidden_size - 1

    print(f"Sample size {sample_size}")
    start = time.time()
    ellipse = Ellipse.from_data(logprobs[:sample_size], hidden_size, verbose=True)
    seconds = time.time() - start
    print(f"Took {seconds} seconds")
    npz_file = (
        f"data/narrow_band_ellipse_pred_{sample_size}_samples.npz"
        if narrow_band
        else f"data/ellipse_pred_{sample_size}_samples.npz"
    )
    np.savez(npz_file, **asdict(ellipse), time=seconds)

if __name__ == "__main__":
    fire.Fire(main)
