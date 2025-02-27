import itertools as it, sys, time
import numpy as np
import scipy
from ellipse_attack.transformations import Ellipse

narrow_band = "--filter" in sys.argv
hidden_size = 64
fname = (
    "data/single_token_prompts/narrow_band_logprobs.npy"
    if narrow_band
    else "data/single_token_prompts/logprobs.npy"
)
logprobs = np.load(fname)

ellipse_rank = hidden_size - 1

sample_sizes = [5000, 10_000, 20_000, 30_000, 100_000]
for sample_size in (
    *it.takewhile(lambda n: n <= logits.shape[0], sample_sizes),
    None,
):
    print(f"Sample size {sample_size}")
    start = time.time()
    ellipse = Ellipse.from_data(logprobs, hidden_size, verbose=True)
    seconds = time.time() - start
    print(f"Took {seconds} seconds")
    outfile = "data/narrow_band_times.dat" if narrow_band else "data/times.dat"
    with open(outfile, "a") as times:
        print(sample_size, seconds, file=times)
    npz_file = (
        f"data/narrow_band_ellipse_pred_{sample_size}_samples.npz"
        if narrow_band
        else f"data/ellipse_pred_{sample_size}_samples.npz"
    )
    np.savez(npz_file, **ellipse)
