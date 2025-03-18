import itertools as it, sys, time
from dataclasses import asdict
import numpy as np
import scipy
from ellipse_attack.transformations import Ellipse, Model
import fire


def main(sample_size: int, infile, outfile, narrow_band: bool = False, wide_band: bool = False, down_proj=None, seed=None, random=False):
    rng = np.random.default_rng(seed)
    hidden = np.load(infile)["hidden"]
    hidden_size = hidden.shape[-1]
    ellipse_rank = hidden_size - 1
    print(f"Sample size {sample_size}")
    start = time.time()
    if down_proj is not None:
        down_proj = np.load(down_proj)
    hidden_subset = (
            hidden[:sample_size] if not random 
            else rng.choice(hidden, size=sample_size, replace=False)
            )
    model = Model(**np.load("data/model/TinyStories-1M.npz"))
    logprobs = scipy.special.log_softmax(hidden_subset @ model.unembed, axis=-1)
    ellipse = Ellipse.from_data(
            logprobs, hidden_size, down_proj=down_proj, verbose=True
            )
    seconds = time.time() - start
    print(f"Took {seconds} seconds")
    np.savez(outfile, **asdict(ellipse), time=seconds, down_proj=down_proj)

if __name__ == "__main__":
    fire.Fire(main)
