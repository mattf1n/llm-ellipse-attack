import itertools as it, functools as ft, operator as op, time, os
from dataclasses import asdict
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import fire
from jaxtyping import Float, Array, Int
from typing import Iterable

from ellipse_attack.transformations import Model


def batched(iterable, n, strict=False):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    seq = iter(iterable)
    while batch := tuple(it.islice(seq, n)):
        if len(batch) == n or not strict:
            yield batch


@torch.inference_mode()
def inference(model, input_ids: Iterable[Iterable[int]], batch_size=1000):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    batches: Iterable[tuple[tuple[int]]] = batched(input_ids, batch_size)
    logit_batches: list[Float[Array, "batch seq vocab"]] = []
    hidden_batches: list[Float[Array, "batch seq hidden"]] = []
    prenorm_batches: list[Float[Array, "batch seq hidden"]] = []
    model.to(device)
    model.eval()
    for batch in tqdm(batches, desc="Running inference"):
        batch_tensor: Num[Array, "batch seq"] = torch.tensor(
            batch, device=device
        )
        output = model(batch_tensor, output_hidden_states=True)
        logit_batches.append(output.logits.cpu().numpy())
        hidden_batches.append(output.hidden_states[-1].cpu().numpy())
        prenorm_batches.append(output.hidden_states[-2].cpu().numpy())
    logits: Float[Array, "doc*seq vocab"] = np.vstack(logit_batches).reshape(
        -1, model.config.vocab_size
    )
    hiddens: Float[Array, "doc*seq hidden"] = np.vstack(hidden_batches).reshape(
        -1, model.config.hidden_size
    )
    prenorms: Float[Array, "doc*seq hidden"] = np.vstack(
        prenorm_batches
    ).reshape(-1, model.config.hidden_size)
    return logits, hiddens, prenorms


@torch.inference_mode()
def main(dataset=None, batch_size=1000, model_name="roneneldan/TinyStories-1M"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if dataset is None:
        input_ids: Int[Array, "vocab 1"] = torch.arange(
            model.config.vocab_size
        )[:, None]
    else:
        data = load_dataset(dataset, streaming=True, trust_remote_code=True)[
            "train"
        ]
        tokenized = iter(data.map(tokenizer, input_columns="text"))
        input_id_seqs: Iterable[list[int]] = map(
            op.itemgetter("input_ids"), tokenized
        )
        input_id_stream = it.chain.from_iterable(input_id_seqs)
        seq_len = 512
        collated_seq_stream: Iterable[tuple[int]] = batched(
            input_id_stream, seq_len, strict=True
        )
        test_sample = next(collated_seq_stream)
        assert len(test_sample) == 512
        input_ids: Iterable[tuple[int]] = it.islice(collated_seq_stream, 100)
    logits, hidden, prenorm = inference(model, input_ids, batch_size=batch_size)
    datasetname = "single_token_prompts" if dataset is None else os.path.basename(dataset)
    dirname = os.path.join(datasetname, os.path.basename(model_name))
    os.makedirs(os.path.join("data", dirname), exist_ok=True)
    np.savez(
        f"data/{dirname}/outputs.npz",
        logits=logits,
        hidden=hidden,
        prenorm=prenorm,
    )


if __name__ == "__main__":
    fire.Fire(main)
