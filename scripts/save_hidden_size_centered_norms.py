import numpy as np

data = np.load("data/model_params.npz")
print(data)
hidden_states = (data["hidden"] - data["beta"]) / data["gamma"]
print(hidden_states.shape)
norms = np.linalg.norm(hidden_states - hidden_states.mean(axis=1, keepdims=True), axis=1)
with open("data/norms.dat", "w") as file:
    for norm in norms:
        print(norm, file=file)


pre_standard_norms = np.sqrt(
        norms ** 2 * 1e-5 
        / (1 - (norms ** 2 / hidden_states.shape[1]))
        )

with open("data/pre_std_norms.dat", "w") as file:
    for norm in pre_standard_norms:
        print(norm, file=file)


"""
n**2 * epsilon / (1 - (n**2 / hidden_size)) = sum(x**2)
"""


var = 0.001
print(8 * (var / (var + 1e-5)))
