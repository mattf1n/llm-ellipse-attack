import scipy
import numpy as np

# def test_rank_of_logits():
#     outputs = np.load("data/single_token_prompts/outputs.npz")
#     true_rank = outputs["hidden"].shape[1]
#     assert np.linalg.matrix_rank(outputs["logits"]) == true_rank
#     logprobs = scipy.special.log_softmax(outputs["logits"], axis=-1)
#     assert np.linalg.matrix_rank(logprobs) == true_rank
#     assert np.linalg.matrix_rank(logprobs - logprobs[0]) == true_rank - 1

def test_rank_of_narrow_band_logits():
    outputs = np.load("data/narrow_band_logits.npz")
    true_rank = 64
    print("Getting log softmax")
    logprobs = scipy.special.log_softmax(outputs["logits"], axis=-1)

    smaller_logprobs = logprobs[:, :true_rank + 1000]
    print("testing rank of smaller logprobs")
    rank = np.linalg.matrix_rank(smaller_logprobs)
    assert rank == true_rank, f"{rank}, {true_rank}"
    print("testing unbiased rank of smaller logprobs")
    rank = np.linalg.matrix_rank(smaller_logprobs - smaller_logprobs[0])
    assert rank == true_rank - 1, f"{rank}, {true_rank-1}"

    print("testing rank")
    rank = np.linalg.matrix_rank(logprobs)
    assert rank == true_rank, f"{rank}, {true_rank}"
    print("testing unbiased rank")
    rank = np.linalg.matrix_rank(logprobs - logprobs[0])
    assert rank == true_rank - 1, f"{rank}, {true_rank-1}"


if __name__ == "__main__":
    test_rank_of_narrow_band_logits()
