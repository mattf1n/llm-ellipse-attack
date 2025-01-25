from transformers import AutoModelForCausalLM
import cvxpy as cp
import numpy as np

np.random.seed(1)

W = (
    AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")
    .get_output_embeddings()
    .weight.detach()
    .numpy()
)

vocab, emb = W.shape

x = cp.Variable(emb)
problem = cp.Problem(cp.Maximize(cp.exp((W @ x)[0] - cp.log_sum_exp(W @ x))))
soln = problem.solve(qcp=True, solver="ECOS", verbose=True)
print(soln)
print(W @ x.value)
