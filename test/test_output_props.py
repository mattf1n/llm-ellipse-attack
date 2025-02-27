import scipy
import numpy as np

# def test_rank_of_logits():
#     outputs = np.load("data/single_token_prompts/outputs.npz")
#     true_rank = outputs["hidden"].shape[1]
#     assert np.linalg.matrix_rank(outputs["logits"]) == true_rank
#     logprobs = scipy.special.log_softmax(outputs["logits"], axis=-1)
#     assert np.linalg.matrix_rank(logprobs) == true_rank
#     assert np.linalg.matrix_rank(logprobs - logprobs[0]) == true_rank - 1

def test_rank_of_logprobs():
    rank = np.linalg.matrix_rank(logprobs)
    assert rank == true_rank, f"{rank}, {true_rank}"

def test_rank_of_unbiased_logprobs():
    rank = np.linalg.matrix_rank(logprobs - logprobs[0])
    assert rank == true_rank - 1, f"{rank}, {true_rank-1}"

true_rank = 64
logprobs = np.load("data/single_token_prompts/logprobs.npy")[:pow(true_rank, 2)]
