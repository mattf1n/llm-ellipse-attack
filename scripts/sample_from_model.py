import itertools, time
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
from get_ellipse import get_ellipse

def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch :.= tuple(itertools.islice(it, n)):
        yield batch


batch_size = 1000
device = (
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
        )
with torch.inference_mode():
    tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-1M")
    model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")
    model.to("mps")
    model.eval()
    print(model)
    W = model.lm_head.weight.cpu().numpy().T
    gamma = model.transformer.ln_f.weight.cpu().numpy()
    beta = model.transformer.ln_f.bias.cpu().numpy()
    input_ids = np.arange(model.config.vocab_size)[:, None]
    batches = batched(input_ids, batch_size)
    logit_batches = []
    hidden_batches = []
    prenorm_batches = []
    for batch in tqdm(map(np.array, batches)):
        output = model(torch.tensor(batch, device=device), output_hidden_states=True)
        logit_batches.append(output.logits[:, -1, :1000].cpu().numpy())
        hidden_batches.append(output.hidden_states[-1][:, -1, :].cpu().numpy())
        prenorm_batches.append(output.hidden_states[-2][:, -1, :].cpu().numpy())

logits = np.vstack(logit_batches)
hidden = np.vstack(hidden_batches)
prenorm = np.vstack(prenorm_batches)
np.savez(
    "data/model_params.npz",
    W=W,
    gamma=gamma,
    beta=beta,
    logits=logits,
    hidden=hidden,
    prenorm=prenorm,
)

print(logits.shape)
# logits = logits - np.mean(logits, axis=1, keepdims=True)
rank = np.linalg.matrix_rank(logits, tol=1e-3)
print("Rank is", rank)
n = rank - 1

for samples in [5000, 10_000, 20_000, 30_000, None]:
    start = time.time()
    S_pred, U_pred, bias_pred = get_ellipse(logits[:samples, :n])
    seconds = time.time() - start
    with open("data/times.dat", "a") as times:
        print(samples, seconds, file=times)
    np.savez(
        f"data/ellipse_pred_{samples}_samples.npz",
        S_pred=S_pred,
        U_pred=U_pred,
        bias_pred=bias_pred,
    )
