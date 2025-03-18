"""
The goal of this script is to investigate whether an approximate ellipse learned from model outputs 
can be used to identify that model's outputs.

Can it differentiate between model outputs and
- random vectors - almost certainly
- random vectors on the unembed image - probably?
- other model outputs mapped to the image - hardest, but probably still

Outputs will generally *not* be on the ellipse.
What we can do is measure distance to the ellipse.
Model outputs will be close to the ellipse surface, or well within it, 
while non-model outputs will be far from the ellipse surface.

If the distribution of errors is known (or estimated), then we can do hypothesis testing.

Let's begin by plotting the errors from model outputs.
"""

import numpy as np, scipy
import matplotlib.pyplot as plt
from ellipse_attack.transformations import Ellipse, Model, alr
import fire

def main(logprobs_file="data/pile-uncopyrighted/logprobs.npy"):
    print("Loading ellipse")
    ellipse = Ellipse.from_npz("data/pile-uncopyrighted/ellipse_pred/20000_samples.npz")
    print("Loading logprobs")
    logprobs = np.load(logprobs_file)[20_000:21_000]
    print("Loading hidden states")
    hidden_states_other = np.load("data/pile-uncopyrighted/TinyStories-3M/disguised_logprobs.npy")[20_000:21_000]
    print("Loading model")
    target_model = Model(**np.load("data/model/TinyStories-1M.npz"))
    print("Computing other logprobs")
    logprobs_other = scipy.special.log_softmax(hidden_states_other @ target_model.unembed, axis=-1)
    print("Computing error")
    errors = ellipse.error(logprobs)
    print("Computing other error")
    errors_other = ellipse.error(logprobs_other)
    np.savez("data/pile-uncopyrighted/ellipse_violations.npz", errors=errors, errors_other=errors_other)
    plt.hist(errors)
    plt.hist(errors_other)
    plt.savefig("results/errors.pdf")

if __name__ == "__main__":
    fire.Fire(main)
