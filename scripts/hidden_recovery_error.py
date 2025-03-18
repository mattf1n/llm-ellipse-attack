from ellipse_attack.transformations import Ellipse, Model
import scipy
import numpy as np

print("Loading hidden states")
hidden = np.load("data/pile-uncopyrighted/TinyStories-1M/outputs.npz")["hidden"]
test = hidden[20_000:21_000]
basis = hidden[21_000:21_064]

print("Loading ellipse")
ellipse_pred = Ellipse.from_npz("data/pile-uncopyrighted/ellipse_pred/20000_samples.npz")

print("Loading model")
model = Model(**np.load("data/model/TinyStories-1M.npz"))
ellipse = model.ellipse()

print("Calculating logprobs")
logprobs = scipy.special.log_softmax(test @ model.unembed, axis=-1)

print("Calculating basis logprobs")
basis_logprobs = scipy.special.log_softmax(basis @ model.unembed, axis=-1)

print("SVD of basis", basis_logprobs.shape)
U, S, Vh = np.linalg.svd(basis_logprobs - basis_logprobs.mean(axis=-1, keepdims=True), full_matrices=False)

print("pinv of basis")
down_proj = np.linalg.pinv(Vh)
baseline_inverted_pred = (logprobs - logprobs.mean(axis=-1, keepdims=True)) @ down_proj

print("Inverting logprobs")
inverted_pred = ellipse_pred.inv(logprobs)
inverted = ellipse.inv(logprobs)

# Our goal is to minimize inverted_pred @ Ω = inverted for a rotation matrix Ω
# This is the Orthogonal Procrustes problem, for which the solution
# is U @ Vh where U @ Σ @ Vh = inverted_pred.T @ inverted


def procrustes(A, B):
    U, S, Vh = np.linalg.svd(A.T @ B)
    return U @ Vh

print("Solving the Orthogonal Procrustes problem")
baseline = procrustes(baseline_inverted_pred, inverted)
ours = procrustes(inverted_pred, inverted)

print("Ours", np.linalg.norm(inverted_pred @ ours - inverted))
print("Baseline", np.linalg.norm(inverted_pred @ baseline - inverted))

