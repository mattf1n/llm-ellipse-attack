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

import numpy as np
import gnuplotlib as gp
from ellipse_attack.transformations import Ellipse, alr


ellipse = Ellipse.from_npz("data/pile-uncopyrighted/ellipse_pred/20000_samples.npz")
logprobs = np.load("data/pile-uncopyrighted/logprobs.npy")
logits = alr(data)
errors = ellipse.error(logits)
plt.hist(errors)
