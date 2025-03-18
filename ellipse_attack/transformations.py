import functools as ft, sys
from dataclasses import dataclass
import numpy as np
from scipy.special import log_softmax, logsumexp
from jaxtyping import Num, Array, Float, Int
from ellipse_attack.get_ellipse import get_ellipse


@dataclass
class Ellipse:
    up_proj: Num[Array, "emb-1 vocab-1"]
    bias: Num[Array, "vocab-1"]
    rot1: None | Num[Array, "emb-1 emb-1"]
    stretch: Num[Array, "emb-1"]
    rot2: Num[Array, "emb-1 emb-1"]

    def __call__(
        self, sphere: Num[Array, "... emb"]
    ) -> Num[Array, "... vocab"]:
        rot1 = np.eye(self.emb_size - 1) if self.rot1 is None else self.rot1
        return alrinv(
            sphere
            @ isom(self.emb_size)
            @ rot1
            @ np.diag(self.stretch)
            @ self.rot2
            @ self.up_proj
            + self.bias
        )

    def inv(self, logprobs: Num[Array, "... vocab"]) -> Num[Array, "... emb"]:
        rot1 = np.eye(self.emb_size - 1) if self.rot1 is None else self.rot1
        return (
            (alr(logprobs) - self.bias)
            @ np.linalg.pinv(self.up_proj)
            @ np.linalg.inv(self.rot2)
            @ np.linalg.inv(np.diag(self.stretch))
            @ np.linalg.inv(rot1)
            @ isom_inv(self.emb_size)
        )

    @classmethod
    def from_data(
        cls,
        logprobs: Num[Array, "sample vocab"],
        emb_size: int,
        down_proj: None | Num[Array, "vocab-1 emb-1"] = None,
        **kwargs
    ):
        vocab_size = logprobs.shape[1]
        print("Computing ALR", file=sys.stderr)
        alr_logits = alr(logprobs)
        if down_proj is None:
            down_proj = np.eye(vocab_size - 1, emb_size - 1)
        unbiased_alr_logits = alr_logits - alr_logits[0]
        full_rank_ellipse = unbiased_alr_logits @ down_proj
        print("Computing up-projection", file=sys.stderr)
        up_proj = (
            np.linalg.pinv(full_rank_ellipse[: emb_size * 2])
            @ unbiased_alr_logits[: emb_size * 2]
        )
        print("Computing ellipse", file=sys.stderr)
        _, stretch, rot2, bias = get_ellipse(full_rank_ellipse, **kwargs)
        return cls(
            up_proj=up_proj,
            bias=bias @ up_proj + alr_logits[0],
            rot1=None,
            stretch=stretch,
            rot2=rot2,
        )

    @classmethod
    def from_npz(cls, fname):
        params = np.load(fname, allow_pickle=True)
        return cls.from_dict(params)

    @classmethod
    def from_dict(cls, params):
        rot1 = None if params["rot1"] == np.array(None) else params["rot1"]
        return cls(
            up_proj=params["up_proj"],
            bias=params["bias"],
            rot1=rot1,
            stretch=params["stretch"],
            rot2=params["rot2"],
        )

    def error(self, logprobs: Num[Array, "... vocab"]) -> Num[Array, "..."]:
        sphere_points = self.inv(logprobs)
        return np.linalg.norm(sphere_points, axis=-1) - np.sqrt(self.emb_size)

    @property
    def emb_size(self):
        return len(self.stretch) + 1


@dataclass
class Model:
    stretch: Num[Array, "emb"]
    bias: Num[Array, "emb"]
    unembed: Num[Array, "emb vocab"]

    def __call__(self, sphere):
        return log_softmax(
            (sphere * self.stretch + self.bias) @ self.unembed, axis=-1
        )

    def ellipse(self, down_proj=None):
        emb_size, vocab_size = self.unembed.shape
        linear = alr(log_softmax(np.diag(self.stretch) @ self.unembed, axis=-1))
        if down_proj is None:
            down_proj = np.eye(vocab_size - 1, emb_size - 1)
        out_basis = center(emb_size) @ linear
        up_proj = np.linalg.pinv(out_basis @ down_proj) @ out_basis
        rot1, stretch, rot2 = np.linalg.svd(
            isom_inv(emb_size) @ linear @ down_proj, full_matrices=False
        )
        # Ensure leading entries of rot2 are always positive
        signs = (rot2[:, [0]] >= 0) * 2 - 1
        rot1, rot2 = signs.T * rot1, signs * rot2
        bias = alr(log_softmax(self.bias @ self.unembed, axis=-1))
        return Ellipse(
            up_proj=up_proj,
            bias=bias,
            rot1=rot1,
            stretch=stretch,
            rot2=rot2,
        )


def ctr_to_alr(n: int) -> Num[Array, "N N-1"]:
    return np.linalg.pinv(center(n)) @ alr(log_softmax(center(n), axis=-1))


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


def alr(x: Num[Array, "... N"]) -> Num[Array, "... N-1"]:
    return x[..., 1:] - x[..., [0]]


def alrinv(x: Num[Array, "... N"]) -> Num[Array, "... N+1"]:
    y0 = -logsumexp(x, axis=-1, keepdims=True)
    return np.concatenate([y0, x + y0], axis=-1)


def isom(n: int):
    return get_transform(np.ones(n), one_hot(n, 0))[:, 1:]


def center(n: int):
    return np.eye(n) - 1 / n


def isom_inv(n: int):
    return np.linalg.pinv(center(n) @ isom(n)) @ center(n)

def angle_error(ellipse: Ellipse, model: Model, logprobs: Float[Array, "... vocab"]):
    ...

def logprobs_of_hidden(model: Model, hidden: Float[Array, "... emb"]) -> Float[Array, "... vocab"]:
    return log_softmax(hidden @ model.embed)
