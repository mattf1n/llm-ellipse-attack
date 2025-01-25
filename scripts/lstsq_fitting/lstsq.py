import numpy as np
import jax
import matplotlib.pyplot as plt


def reflect(A, n):
    return A - 2 * np.outer(n, (n @ A) / (n @ n))


def rotate(u, v):
    S = reflect(np.eye(len(u)), u + v)
    return reflect(S, v)


dim = 4
emb = dim + 1
vocab = emb * 2
batch = 50
gamma, beta = np.random.rand(emb), np.random.rand(emb)
unembed = np.random.rand(emb, vocab)
down_proj = np.eye(vocab, dim)
center = np.eye(vocab) - np.eye(vocab).mean(0)
u = np.ones(emb) / np.linalg.norm(np.ones(emb))
v = np.arange(emb) == 0
S = reflect(np.eye(emb), u + v)
R = reflect(S, v)[1:]  # Dim, Emb

h = np.random.rand(batch, emb)
sphere = (jax.nn.standardize(h) / np.sqrt(emb)) @ R.T
logits = (jax.nn.standardize(h) * gamma + beta) @ unembed  # Batch, Vocab
# logits = jax.nn.standardize(h) @ np.diag(gamma) @ unembed + beta @ unembed
recovered_logits = logits - np.mean(logits, axis=1, keepdims=True)  # Batch, Vocab
x = recovered_logits @ down_proj  # Batch, Dim
print(x.shape)
# plt.scatter(x[:, 0], x[:, 1])
# plt.show()


A = (
    np.sqrt(emb) * np.linalg.pinv(R.T) @ np.diag(gamma) @ unembed @ center @ down_proj
)  # Emb, Dim
b = beta @ unembed @ center @ down_proj  # Vocab
print((sphere @ A + b)[:4])
print(x[:4])


def get_second_order_terms(x):
    outer_prod = x.reshape(batch, dim, 1) * x.reshape(batch, 1, dim)
    return (outer_prod).reshape(batch, dim * dim)


X_ = np.concatenate((get_second_order_terms(x), x), axis=1)
Cb = np.linalg.lstsq(X_, np.ones(batch), rcond=None)[0]
C_ = Cb[: dim * dim].reshape(dim, dim)
b_ = np.linalg.inv(-2 * C_) @ Cb[-dim:]  # Dim
print("Center", b)
print("Center pred", b_)

# center data
X = get_second_order_terms(x - b_)
C = np.linalg.lstsq(X, np.ones(batch), rcond=None)[0].reshape(dim, dim)
A_ = np.linalg.cholesky(np.linalg.inv(C)).T
A_svd = np.linalg.svd(A_)
A__ = A_svd.U.T @ A_

print(jax.vmap(lambda x: (x - b) @ np.linalg.inv(A.T @ A) @ (x - b).T)(x))
print(jax.vmap(lambda x: (x - b) @ np.linalg.inv(A_.T @ A_) @ (x - b).T)(x))
print(jax.vmap(lambda x: (x - b) @ np.linalg.inv(A__.T @ A__) @ (x - b).T)(x))

print("A,A_,A__")
print(A)
print(A_)
print(A__)

print("(AA^T)^-1")
print(C)
print(np.linalg.inv(A.T @ A))
print(np.linalg.inv(A_.T @ A_))
print(np.linalg.inv(A__.T @ A__))

print("Singular values")
print(np.linalg.svd(A).S)
print(np.linalg.svd(A_).S)
print(np.linalg.svd(A__).S)

print("Rotations")
print(np.linalg.svd(A).Vh)
print(np.linalg.svd(A_).Vh)
print(np.linalg.svd(A__).Vh)
