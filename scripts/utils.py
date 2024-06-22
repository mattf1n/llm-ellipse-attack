import itertools as it, math, argparse
import torch

def get_second_order_terms(x):
    outer_prod = x[:, :, None] * x[:, None, :]
    indices = torch.triu_indices(*outer_prod.shape[1:])
    return outer_prod[:, *indices]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="rand")
    parser.add_argument("--device", default=None)
    return parser.parse_args()

def get_device(device):
    return (
        device if device is not None 
        else "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
