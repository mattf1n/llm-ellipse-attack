import functools as ft, itertools as it, os
import numpy as np
from scipy.special import log_softmax
import scipy
from transformers import AutoTokenizer
from sklearn.mixture import BayesianGaussianMixture
import fire
from tqdm import tqdm
from ellipse_attack.transformations import Model


def main(dataset="vocab"):
    tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-1M")

    data = np.load(f"data/{dataset}/model_params.npz")
    final_layer = Model(**np.load(f"data/model/TinyStories-1M.npz"))
    outdir = os.path.join("overleaf", "data", dataset)
    os.makedirs(outdir, exist_ok=True)
    print(data)
    hidden_states = (data["hidden"] - final_layer.bias) / final_layer.stretch
    print(hidden_states.shape)

    def save_values_to_file(fname, *values):
        with open(os.path.join(outdir, fname), "w") as file:
            for vals in tqdm(
                zip(*values), total=len(values[0]), desc=f"Writing {fname}"
            ):
                print(*vals, sep="\t", file=file)

    norms = np.linalg.norm(hidden_states, axis=1)
    save_values_to_file("norms.dat", norms)

    def save_gaussian_mixture_component_pdfs(filename, gm, data):
        domain = np.linspace(min(data), max(data), 100)
        weights = gm.weights_
        means = gm.means_.squeeze()
        stds = np.sqrt(gm.covariances_.squeeze().squeeze())
        gauss1 = weights[0] * scipy.stats.norm.pdf(domain, means[0], stds[0])
        gauss2 = weights[1] * scipy.stats.norm.pdf(domain, means[1], stds[1])
        save_values_to_file(filename, domain, gauss1, gauss2, gauss1 + gauss2)

    # Split into train and test tokens
    rng = np.random.default_rng()
    shuffled_tokens = rng.permutation(np.arange(len(norms)))
    test_size = 10000
    test_tokens, train_tokens = shuffled_tokens[:test_size], shuffled_tokens[test_size:]

    greedy_next_token_ids = data["logits"].argmax(axis=1)

    print("Computing entropies")
    all_entropies = -log_softmax(data["logits"], axis=1).mean(axis=1)
    test_entropies, entropies = all_entropies[test_tokens], all_entropies[train_tokens]
    save_values_to_file("entropies.dat", entropies)
    entropy_gm = BayesianGaussianMixture(n_components=2, tol=1e-5, random_state=0).fit(
        entropies.reshape(-1, 1)
    )
    save_gaussian_mixture_component_pdfs("entropy_gm_fit.dat", entropy_gm, entropies)

    # sqrt(n**2 * epsilon / (1 - (n**2 / hidden_size))) = norm(x)
    pre_standard_norms = np.sqrt(
        norms**2 * 1e-5 / (1 - (norms**2 / hidden_states.shape[1]))
    )
    save_values_to_file("pre_std_norms.dat", pre_standard_norms)
    gm = BayesianGaussianMixture(n_components=2, tol=1e-5, random_state=2).fit(
        pre_standard_norms.reshape(-1, 1)
    )
    print(f"{gm.converged_=}")
    save_gaussian_mixture_component_pdfs("gauss_fit.dat", gm, pre_standard_norms)

    probs = entropy_gm.predict_proba(test_entropies.reshape(-1, 1))[:, 0]
    save_values_to_file("dist1_probs.dat", probs)
    predictions = probs > 0.90
    save_values_to_file("dist1_norms.dat", norms[test_tokens][predictions])
    save_values_to_file("dist2_norms.dat", norms[test_tokens][~predictions])

    is_dist1 = entropy_gm.predict_proba(all_entropies.reshape(-1, 1))[:,0] > 0.90
    np.savez("data/narrow_band_logits.npz", logits=data["logits"][is_dist1])


if __name__ == "__main__":
    fire.Fire(main)
