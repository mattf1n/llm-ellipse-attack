import argparse, os, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="rand")
    return parser.parse_args()


@torch.inference_mode()
def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    args = get_args()
    model_basename = os.path.basename(args.model)

    if args.model == "test":
        vocab_size = 8
        hidden_size = 4
        sample_size = 50
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
        sample_size = hidden_size * 10
        input_ids = torch.arange(sample_size, device=device).reshape(sample_size, 1)
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
