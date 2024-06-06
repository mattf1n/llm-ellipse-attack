import torch
import matplotlib.pyplot as plt


def main():
    logits = torch.load("data/logits.pt").cpu()
    reparam_bias = torch.load("data/reparam_bias.pt").cpu()
    print(logits.max(axis=1))
    x, y = torch.topk(
        logits.max(axis=0).values - logits.min(axis=0).values, k=2
    ).indices
    x, y = 111, 156
    plt.scatter(logits[:, x], logits[:, y], s=0.1)
    plt.scatter(reparam_bias[x], reparam_bias[y])
    plt.show()


if __name__ == "__main__":
    main()
