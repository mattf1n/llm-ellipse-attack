import itertools as it, sys, time
from dataclasses import asdict
import numpy as np
import scipy
from ellipse_attack.transformations import Ellipse
import fire


def main(sample_size: int, infile, outfile, narrow_band: bool = False, wide_band: bool = False, down_proj=None):
    hidden_size = 64
    logprobs = np.load(infile)

    ellipse_rank = hidden_size - 1

    print(f"Sample size {sample_size}")
    start = time.time()
    if down_proj is not None:
        down_proj = np.load(down_proj)
    ellipse = Ellipse.from_data(
            logprobs[:sample_size], hidden_size, down_proj=down_proj, verbose=True
            )
    seconds = time.time() - start
    print(f"Took {seconds} seconds")
    np.savez(outfile, **asdict(ellipse), time=seconds, down_proj=down_proj)

if __name__ == "__main__":
    fire.Fire(main)
