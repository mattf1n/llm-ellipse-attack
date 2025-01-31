import itertools as it, sys, time
import numpy as np
from get_ellipse import get_ellipse

debug = "--debug" in sys.argv
hidden_size = 64
logits = np.load("data/narrow_band_logits.npz")["logits"]
recovered_logits = logits - np.mean(logits, axis=1, keepdims=True)
print(recovered_logits.shape)

if debug:
    rank = np.linalg.matrix_rank(recovered_logits[:hidden_size+100, :hidden_size+100], tol=1e-3)
    assert (
        rank == hidden_size
    ), f"Rank of logits should equal hidden size. Got {rank=}, {hidden_size=}"

ellipse_rank = hidden_size - 1

sample_sizes = [5000, 10_000, 20_000, 30_000, 100_000]
for samples in (
    *it.takewhile(lambda n: n <= recovered_logits.shape[0], sample_sizes),
    None,
):
    start = time.time()
    C, S_pred, U_pred, bias_pred = get_ellipse(
        recovered_logits[:samples, :ellipse_rank]
    )
    seconds = time.time() - start
    with open("data/narrow_band_times.dat", "a") as times:
        print(samples, seconds, file=times)
    np.savez(
        f"data/narrow_band_ellipse_pred_{samples}_samples.npz",
        S_pred=S_pred,
        U_pred=U_pred,
        bias_pred=bias_pred,
    )
