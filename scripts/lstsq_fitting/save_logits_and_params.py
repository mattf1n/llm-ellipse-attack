import argparse, os, json, itertools as it, math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import get_args, get_device


@torch.inference_mode()
def main():
    args = get_args()
    device = get_device(args.device)
    model_basename = os.path.basename(args.model)

    if args.model == "test":
        vocab_size = 100
        hidden_size = 40
        sample_size = math.comb(hidden_size, 2) + hidden_size - 1
        gamma, beta = torch.rand(hidden_size), torch.rand(hidden_size)
        embeds = torch.rand(vocab_size, hidden_size)
        weight = embeds @ torch.diag(gamma)
        bias = embeds @ beta
        activations = torch.rand(sample_size, hidden_size)
        standardized = (
            activations - activations.mean(axis=1, keepdims=True)
        ) / activations.std(axis=1, keepdims=True)
        raw_logits = standardized @ weight.T + bias
        logprobs = raw_logits - torch.logsumexp(raw_logits, axis=1, keepdims=True)
        logits = logprobs - logprobs.mean(axis=1, keepdims=True)
        logits = raw_logits - raw_logits.mean(axis=1, keepdims=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
        embeds = model.get_output_embeddings().weight
        gamma = model.get_submodule("gpt_neox.final_layer_norm").weight
        beta = model.get_submodule("gpt_neox.final_layer_norm").bias
        weight = embeds @ torch.diag(gamma)
        bias = embeds @ beta
        vocab_size = model.config.vocab_size
        hidden_size = model.config.hidden_size
        sample_size = math.comb(hidden_size, 2) + hidden_size - 1
        print(sample_size)
        input_ids = torch.tensor(
            list(it.islice(it.product(range(vocab_size), repeat=2), sample_size)),
            device=device,
        )
        print(input_ids.shape)
        assert input_ids.shape == (sample_size, 2)
        raw_logits = model(input_ids).logits[:, -1, :]  # shape: (Samp, Vocab)
        logprobs = raw_logits - torch.logsumexp(raw_logits, axis=1, keepdims=True)
        logits = logprobs - logprobs.mean(axis=1, keepdims=True)

    config = dict(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        sample_size=sample_size,
    )
    path = f"data/{model_basename}"
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w") as file:
        json.dump(config, file)
    torch.save(logits, os.path.join(path, "logits.pt"))
    torch.save(weight, os.path.join(path, "weight.pt"))
    torch.save(bias, os.path.join(path, "bias.pt"))


if __name__ == "__main__":
    main()
