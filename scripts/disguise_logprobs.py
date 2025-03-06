import fire
import numpy as np
import jax, jax.numpy as jnp
import jaxopt
from tqdm import tqdm
from ellipse_attack.transformations import alr

jax.config.update("jax_enable_x64", True)

def main(model_name="TinyStories-3M", dataset_name="pile-uncopyrighted", size=None):
    logits = np.load(f"data/{dataset_name}/{model_name}/outputs.npz")["logits"][:size]
    dataset_size, vocab_size = logits.shape
    probs = jax.nn.softmax(logits, axis=-1)
    basis = np.load("data/model/TinyStories-1M.npz")["unembed"]
    embed_size, vocab_size = basis.shape
    hidden = jnp.ones(embed_size)
    solns = []

    @jax.jit
    @jax.value_and_grad
    def loss(hidden, prob):
        logits = hidden @ basis
        logprobs = logits - jax.nn.logsumexp(logits)
        return -(jnp.dot(prob, logprobs)).sum()

    solver = jaxopt.LBFGS(fun=loss, value_and_grad=True, jit=True)

    for prob in tqdm(probs):
        soln, state = solver.run(hidden, prob)
        print(-np.sort(-prob))
        print(-np.sort(-jax.nn.softmax(soln @ basis)))
        solns.append(soln)
    disguised_logprobs = np.vstack(solns)
    np.save(f"data/{dataset_name}/{model_name}/disguised_logprobs.py", disguised_logprobs)


if __name__ == "__main__":
    fire.Fire(main)


