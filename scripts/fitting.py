import functools as ft
import numpy as np
import jax, jax.numpy as jnp
from jax.scipy import optimize as opt
import matplotlib.pyplot as plt


def loss(
    params: jax.Array,  # 2 * Emb
    x: jax.Array,  # Batch, Emb
) -> float:
    error = jax.vmap(ft.partial(ellipsoid, params))(x) - 1  # Batch
    return jnp.square(error).mean()


def ellipsoid(
    params: jax.Array,  # 2 * Emb
    x: jax.Array,  # Emb
) -> float:
    offset, scale = params[: len(x)], params[len(x) :]
    return jnp.square((x - offset) / scale).sum()


emb = 3
gamma, beta = np.random.random(emb), np.random.random(emb)
random_proj = np.random.random((emb, emb - 1))
x = (jax.nn.standardize(np.random.random((10, emb))) * gamma + beta) @ random_proj
plt.scatter(x[:, 0], x[:, 1])
plt.show()
guess = jnp.concatenate((x.mean(0), np.random.random(x.shape[-1])))
assert len(guess) == len(x[-1]) * 2
res = opt.minimize(
    fun=loss, x0=guess, args=(x,), method="BFGS", options={"maxiter": None}
)

print(res)
