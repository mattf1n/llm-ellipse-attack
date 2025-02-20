import functools as ft
from dataclasses import dataclass
import numpy as np
from jaxtyping import Num, Array
from ellipse_attack.get_ellipse import get_ellipse


@dataclass
class Ellipse:
    up_proj: Num[Array, "emb-1 vocab"]
    bias: Num[Array, "vocab"]
    rot1: None | Num[Array, "emb-1 emb-1"]
    stretch: Num[Array, "emb-1"]
    rot2: Num[Array, "emb-1 emb-1"]

    def __call__(self, sphere: Num[Array, "emb"]):
        emb_size = self.stretch.shape[0] + 1
        rot1 = np.eye(emb_size - 1) if self.rot1 is None else self.rot1
        return (
            sphere
            @ isom(emb_size)
            @ rot1
            @ np.diag(self.stretch)
            @ self.rot2
            @ self.up_proj
            + self.bias
        )

    @classmethod
    def from_data(
        cls, logprobs: Num[Array, "sample vocab"], emb_size: int, **kwargs
    ):
        vocab_size = logprobs.shape[1]
        down_proj = alr_transform(vocab_size)[:, :emb_size-1]
        logits = (logprobs - logprobs[0]) @ down_proj
        up_proj = np.linalg.pinv(logits) @ (logprobs - logprobs[0])
        np.testing.assert_allclose(logits @ up_proj + logprobs[0], logprobs)
        _, stretch, rot2, bias = get_ellipse(logits, **kwargs)
        return cls(
            up_proj=up_proj,
            bias=bias @ up_proj + logprobs[0],
            rot1=None,
            stretch=stretch,
            rot2=rot2,
        )


@dataclass
class Model:
    stretch: Num[Array, "emb"]
    bias: Num[Array, "emb"]
    unembed: Num[Array, "emb vocab"]

    def __call__(self, sphere):
        return (sphere * self.stretch + self.bias) @ self.unembed @ center(self.unembed.shape[1])

    def ellipse(self, down_proj=None):
        emb_size, vocab_size = self.unembed.shape
        linear = np.diag(self.stretch) @ self.unembed @ center(vocab_size)
        if down_proj is None:
            down_proj = np.eye(vocab_size, emb_size - 1)
        std_ctr = center(emb_size) @ linear @ center(vocab_size)
        up_proj = np.linalg.pinv(std_ctr @ down_proj) @ std_ctr
        rot1, stretch, rot2 = np.linalg.svd(
            isom_inv(emb_size) @ linear @ down_proj, full_matrices=False
        )
        # Ensure leading entries of rot2 are always positive
        signs = (rot2[:, [0]] >= 0) * 2 - 1
        rot1, rot2 = signs.T * rot1, signs * rot2
        bias = self.bias @ self.unembed @ center(vocab_size)
        return Ellipse(
            up_proj=up_proj,
            bias=bias,
            rot1=rot1,
            stretch=stretch,
            rot2=rot2,
        )


def standardize(x: Num[Array, "... N"]) -> Num[Array, "... N"]:
    return (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(
        x.var(axis=-1, keepdims=True)
    )


def reflect(A, n):
    """Utility for `get_transform`"""
    return A - 2 * np.outer(n, (n @ A) / (n @ n))


def get_transform(u, v):
    """
    Takes two vectors $u$ and $v$ and returns a matrix that maps $u$ into $v$
    by rotating about the vector $u\\times v$.
    """
    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)
    u = u / u_norm
    v = v / v_norm
    S = reflect(np.eye(len(u)), u + v)
    R = reflect(S, v) * v_norm / u_norm
    return R.T


def isometric_transform(u: Num[Array, "... N"]) -> Num[Array, "... N-1"]:
    dim = u.shape[-1]
    one_hot_first = np.arange(dim) == 0
    transform = get_transform(np.ones(dim), one_hot_first)
    return u @ transform[:, 1:]


def one_hot(size, n):
    return np.arange(size) == n


def alr_transform(size):
    """ALR transform: subtract first logprob, drop first logprob"""
    return (np.eye(size) - one_hot(size, 0).reshape(-1, 1))[:, 1:]


def isom(n: int):
    return get_transform(np.ones(n), one_hot(n, 0))[:, 1:]


def center(n: int):
    return np.eye(n) - np.ones(n) / n


def isom_inv(n: int):
    return np.linalg.pinv(center(n) @ isom(n)) @ center(n)


