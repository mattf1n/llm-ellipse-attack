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
    new_sample_size = math.comb(hidden_size + 1, 2)
    est_bias = torch.load(os.path.join(path, "est_bias.pt"))
    points = torch.load(os.path.join(path, "points.pt")).cpu()

    # Representation 3: (points - b) @ coeffs3 @ (points - b).T
    print("Getting representation 3")
    centered_terms = get_second_order_terms(points[:new_sample_size] - est_bias)
    coeffs3_triu = torch.linalg.solve(centered_terms, torch.ones(new_sample_size))
    coeffs3 = torch.eye(hidden_size)
    coeffs3[*torch.triu_indices(hidden_size, hidden_size)] = coeffs3_triu
    coeffs3 = (coeffs3 + coeffs3.T) / 2
    torch.save(coeffs3, os.path.join(path, "coeffs3.pt"))


if __name__ == "__main__":
    main()
