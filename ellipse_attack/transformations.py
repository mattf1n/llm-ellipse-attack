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
            @ isometric_transform_matrix(emb_size)
            @ rot1
            @ np.diag(self.stretch)
            @ self.rot2
            @ self.up_proj
            + self.bias
        )

    @property
    def up_proj_inv(self):
        return np.linalg.pinv(self.up_proj)

    @classmethod
    def from_data(
        cls, logprobs: Num[Array, "sample vocab"], emb_size=None, **kwargs
    ):
        if emb_size is None:
            emb_size = logprobs.shape[1]
        # ALR transform: subtract first logprob, drop first logprob
        logits = (logprobs - logprobs[:, [0]])[:, 1:emb_size]
        up_proj = np.linalg.pinv(logits) @ logprobs
        _, rot2, stretch, bias = get_ellipse(logits, **kwargs)
        return cls(
            up_proj=up_proj,
            bias=bias,
            rot1=None,
            stretch=np.diag(stretch),
            rot2=rot2,
        )


@dataclass
class Model:
    stretch: Num[Array, "emb"]
    bias: Num[Array, "emb"]
    unembed: Num[Array, "emb vocab"]

    def __call__(self, sphere):
        return (sphere * self.stretch + self.bias) @ self.unembed @ self.center

    @property
    def center(self) -> Num[Array, "vocab vocab"]:
        return centering_matrix(self.unembed.shape[1])

    def ellipse(self, up_proj_inv=None):
        bias = self.bias @ self.unembed @ self.center
        emb_size, vocab_size = self.unembed.shape
        linear = np.diag(self.stretch) @ self.unembed @ self.center
        if up_proj_inv is None:
            up_proj_inv = np.eye(vocab_size, emb_size - 1)
        std_ctr = centering_matrix(emb_size) @ linear
        up_proj = np.linalg.pinv(std_ctr @ up_proj_inv) @ std_ctr
        isom_inv: Num[Array, "emb-1 emb"] = isometric_transform_inverse(emb_size)
        rot1, stretch, rot2 = np.linalg.svd(
            isom_inv @ linear @ up_proj_inv, full_matrices=False
        )
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


def _isometric_transform_matrix_square(n: int):
    one_hot_first = np.arange(n) == 0
    transform = get_transform(np.ones(n), one_hot_first)
    return transform

def isometric_transform_matrix(n: int):
    return _isometric_transform_matrix_square(n)[:, 1:]

def isometric_transform_inverse(n: int):
    centered = centering_matrix(n)
    isom = isometric_transform_matrix(n) 
    isom_inv = np.linalg.pinv(centered @ isom) @ centered
    return isom_inv

def centering_matrix(n: int):
    return np.eye(n) - np.ones(n) / n

