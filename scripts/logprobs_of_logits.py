import sys
import scipy, numpy as np

outputs = np.load(sys.stdin.buffer)
logprobs = scipy.special.log_softmax(outputs["logits"], axis=-1)
np.save(sys.stdout.buffer, logprobs)
