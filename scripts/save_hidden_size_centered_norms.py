import numpy as np

data = np.load("data/model_params.npz")
print(data)
hidden_states = (data["hidden"] - data["beta"]) / data["gamma"]
print(hidden_states.shape)
norms = np.linalg.norm(hidden_states - hidden_states.mean(axis=1, keepdims=True), axis=1)
with open("overleaf/data/norms.dat", "w") as file:
    for norm in norms:
        print(norm, file=file)

var = 0.001
print(8 * (var / (var + 1e-5)))
