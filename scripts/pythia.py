import argparse, os, json, math
import torch
from transformers import AutoConfig
from utils import get_second_order_terms


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="rand")
    return parser.parse_args()


@torch.inference_mode()
def main():
    args = get_args()
    model_basename = os.path.basename(args.model)
    path = f"data/{model_basename}"
    with open(os.path.join(path, "config.json")) as file:
        config = json.load(file)
    hidden_size = config["hidden_size"] - 1
    vocab_size = config["vocab_size"]
    sample_size = config["sample_size"]
    coeffs3 = torch.load(os.path.join(path, "coeffs3.pt")).cpu()
    weight = torch.load(os.path.join(path, "weight.pt")).cpu()

    # Representation 4: (points - b) @ (AA^T)^-1 @ (points - b) = 1
    print("Getting representation 4")
    A = torch.linalg.cholesky(torch.linalg.inv(coeffs3)).T
    A_svd = torch.linalg.svd(A).S

    center = torch.eye(vocab_size) - torch.eye(vocab_size).mean(0)
    down_proj = torch.eye(vocab_size, hidden_size)
    u = torch.ones(hidden_size + 1) / torch.linalg.norm(torch.ones(hidden_size + 1))
    v = (torch.arange(hidden_size + 1) == 0).to(A.dtype)
    S = reflect(torch.eye(hidden_size + 1), u + v)
    R = reflect(S, v)[1:]  # Dim, Emb
    R_inv = torch.linalg.pinv(R.T)
    W_svd = torch.linalg.svd(
        math.sqrt(hidden_size) * R_inv @ weight.T @ center @ down_proj
    ).S

    print(A_svd)
    print(W_svd)


def reflect(A, n):
    return A - 2 * torch.outer(n, (n @ A) / (n @ n))


if __name__ == "__main__":
    main()
