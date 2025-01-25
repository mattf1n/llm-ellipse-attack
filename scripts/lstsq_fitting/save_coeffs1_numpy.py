import argparse, operator, os, json
import numpy as np
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
    logits = torch.load(os.path.join(path, "logits.pt")).cpu().numpy()

    down_proj = np.eye(vocab_size, hidden_size)
    points = logits @ down_proj  # shape: (Samp, Hidden - 1)
    terms = np.concatenate((get_second_order_terms(points), points), axis=1)

    # Representation 1: terms @ coeffs1 = 2
    print("Getting representation 1")
    coeffs1 = np.linalg.lstsq(terms, np.ones(sample_size), rcond=None)[0]

    torch.save(points, os.path.join(path, "points.pt"))
    torch.save(coeffs1, os.path.join(path, "coeffs1.pt"))


if __name__ == "__main__":
    main()
