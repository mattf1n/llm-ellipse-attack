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
    coeffs1 = torch.load(os.path.join(path, "coeffs1.pt"))
    bias = torch.load(os.path.join(path, "bias.pt")).cpu()

    # Representation 2: points @ coeffs2 @ points.T + bias = 1
    print("Getting representation 2")
    coeffs2 = coeffs1[: hidden_size * hidden_size].reshape(hidden_size, hidden_size)
    est_bias = torch.linalg.inv(-2 * coeffs2) @ coeffs1[-hidden_size:]  # Dim
    center = torch.eye(vocab_size) - torch.eye(vocab_size).mean(0)
    down_proj = torch.eye(vocab_size, hidden_size)
    print(est_bias)
    print(bias @ center @ down_proj)
    assert torch.allclose(bias @ center @ down_proj, est_bias, atol=1e-5)
    torch.save(est_bias, os.path.join(path, "est_bias.pt"))


if __name__ == "__main__":
    main()
