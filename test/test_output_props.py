import scipy
import numpy as np
import ellipse_attack.transformations as tfm


def test_logprob_recoverability():
    vocab_size = logprobs.shape[1]
    down_proj = tfm.alr_transform(vocab_size)[:, :emb_size-1]
    logits = (logprobs - logprobs[0]) @ down_proj
    up_proj = np.linalg.pinv(logits) @ (logprobs - logprobs[0])
    np.testing.assert_allclose(logits @ up_proj + logprobs[0], logprobs)


true_rank = 64
logprobs = np.load("data/single_token_prompts/logprobs.npy")[:200]
