import numpy as np
from scipy.special import log_softmax
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-1M")

data = np.load("data/model_params.npz")
print(data)
hidden_states = (data["hidden"] - data["beta"]) / data["gamma"]
print(hidden_states.shape)
norms = np.linalg.norm(hidden_states - hidden_states.mean(axis=1, keepdims=True), axis=1)
with open("data/norms.dat", "w") as file:
    for norm in norms:
        print(norm, file=file)

greedy_next_token_ids = data["logits"].argmax(axis=1)

entropies = log_softmax(data["logits"], axis=1).mean(axis=1)

def partition_on_mode(norms):
    mode1_token_ids, mode2_token_ids = [], []
    for token_id, norm in enumerate(norms):
        distance_from_mode1_mean = abs(norm - 7.97)
        if distance_from_mode1_mean <= 0.001:
            mode1_token_ids.append(token_id)
        elif distance_from_mode1_mean >= 0.01:
            mode2_token_ids.append(token_id)
    return mode1_token_ids, mode1_token_ids

mode1_token_ids, mode1_token_ids = partition_on_mode(norms)

mode1_entropies = entropies[mode1_token_ids]
mode2_entropies = entropies[mode2_token_ids]
with open("data/mode1_entropies", "w") as file:
    for entropy in mode1_entropies:
        print(entropy, file=file)
with open("data/mode2_entropies", "w") as file:
    for entropy in mode2_entropies:
        print(entropy, file=file)

entropy_selected_norms = norms[np.abs(entropies - mode1_entropies.mean()) <= mode1_entropies.std() * 0.25]
print("Mode 1 entropies std: ", mode1_entropies.std())
with open("data/entropy_selected_norms.dat", "w") as file:
    for norm in entropy_selected_norms:
        print(norm, file=file)

mode1_entropy_selected_token_ids, mode2_entropy_selected_token_ids = partition_on_mode(entropy_selected_norms)


selected_norms = norms[[tok[0] in [" ", ".", ","] for tok in tokenizer.batch_decode(list(range(len(norms))))]]
with open("data/selected_norms.dat", "w") as file:
    for norm in selected_norms:
        print(norm, file=file)

mode1_greedy_next_token_ids = greedy_next_token_ids[mode1_token_ids]
mode2_greedy_next_token_ids = greedy_next_token_ids[mode2_token_ids]

mode1_tokens, mode2_tokens = map(tokenizer.batch_decode, (mode1_token_ids, mode2_token_ids))
mode1_greedy_next_tokens, mode2_greedy_next_tokens = map(tokenizer.batch_decode, (mode1_greedy_next_token_ids, mode2_greedy_next_token_ids))

print("Next tokens only in mode 1")
print(set(mode1_greedy_next_tokens) - set(mode2_greedy_next_tokens))
print("Next tokens only in mode 2")
print(set(mode2_greedy_next_tokens) - set(mode1_greedy_next_tokens))

def save_values_to_file(fname, values):
    with open(fname, "w") as file:
        for value in values:
            print(value, file=file)

save_values_to_file("data/mode1_token_ids.txt", mode1_tokens)
save_values_to_file("data/mode2_token_ids.txt", mode2_tokens)

save_values_to_file("data/mode1_greedy_next_token.txt",mode1_greedy_next_tokens)
save_values_to_file("data/mode2_greedy_next_token.txt",mode2_greedy_next_tokens)

dictionary = set(map(str.strip, open("/usr/share/dict/words")))
mode1_words = set(map(str.strip, mode1_tokens)) & dictionary
mode2_words = set(map(str.strip, mode2_tokens)) & dictionary
print("Words in mode 1:")
print(round(len(mode1_words) / len(mode1_tokens), 3) * 100)
print("Words in mode 2:")
print(round(len(mode2_words) / len(mode2_tokens), 3) * 100)

print("Space-initial words in mode 1:")
print(round(len([tok for tok in mode1_tokens if tok[0] == " "]) / len(mode1_tokens), 3) * 100)
print("Space-initial words in mode 2:")
print(round(len([tok for tok in mode2_tokens if tok[0] == " "]) / len(mode2_tokens), 3) * 100)

print("Space-initial next words in mode 1:")
print(round(len([tok for tok in mode1_greedy_next_tokens if tok[0] in (" ", ".")]) / len(mode1_greedy_next_tokens), 3) * 100)
print("Space-initial next words in mode 2:")
print(round(len([tok for tok in mode2_greedy_next_tokens if tok[0] in (" ", ".")]) / len(mode2_greedy_next_tokens), 3) * 100)


pre_standard_norms = np.sqrt(
        norms ** 2 * 1e-5 
        / (1 - (norms ** 2 / hidden_states.shape[1]))
        )

with open("data/pre_std_norms.dat", "w") as file:
    for norm in pre_standard_norms:
        print(norm, file=file)


"""
n**2 * epsilon / (1 - (n**2 / hidden_size)) = sum(x**2)
"""


var = 0.001
print(8 * (var / (var + 1e-5)))
