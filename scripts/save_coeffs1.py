import argparse, operator, os, json
import torch
from transformers import AutoConfig
from utils import get_second_order_terms, get_args, get_device


@torch.inference_mode()
def main():
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.backends.cuda.is_available()
        else "cpu"
    )
    args = get_args()
    device = get_device(args.device)
    model_basename = os.path.basename(args.model)
    path = f"data/{model_basename}"
    with open(os.path.join(path, "config.json")) as file:
        config = json.load(file)
    hidden_size = config["hidden_size"] - 1
    vocab_size = config["vocab_size"]
    sample_size = config["sample_size"]
    logits = torch.load(os.path.join(path, "logits.pt"), map_location=device)

    down_proj = torch.eye(vocab_size, hidden_size, device=device)
    points = logits @ down_proj  # shape: (Samp, Hidden - 1)
    torch.save(points, os.path.join(path, "points.pt"))

    # Representation 1: terms @ coeffs1 = 1
    print("Getting representation 1")
    terms = torch.cat((get_second_order_terms(points), points), axis=1)
    coeffs1 = torch.linalg.solve(terms, torch.ones(sample_size, device=device))
    torch.save(coeffs1, os.path.join(path, "coeffs1.pt"))


if __name__ == "__main__":
    main()
