import numpy as np, scipy

compressed_logits = np.load("data/pile-uncopyrighted/logits_compressed.npz")
print(compressed_logits["compressed"].shape, compressed_logits["decompressor"].shape) 
logits = compressed_logits["compressed"] @ compressed_logits["decompressor"]
logprobs = scipy.special.log_softmax(logits, axis=-1)
np.save("data/pile-uncopyrighted/logprobs_decompressed.npy", logprobs)


