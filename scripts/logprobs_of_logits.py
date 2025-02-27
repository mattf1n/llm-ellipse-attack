import scipy, numpy as np

outputs = np.load("data/single_token_prompts/outputs.npz")
logprobs = scipy.special.log_softmax(outputs["logits"], axis=-1)
np.save("data/single_token_prompts/logprobs.npy", logprobs)
