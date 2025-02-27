import itertools as it, sys, time
import numpy as np
import scipy
from ellipse_attack.transformations import Ellipse

narrow_band = "--filter" in sys.argv
hidden_size = 64
fname = (
    "data/narrow_band_logits.npz"
    if narrow_band
    else "data/vocab/model_params.npz"
)
logits = np.load(fname)["logits"]
logprobs = scipy.special.log_softmax(logits, axis=-1)

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
    np.savez(f"data/narrow_band_ellipse_pred_{sample_size}_samples.npz", **ellipse)
