import numpy as np
import fire

def main(in_size, out_size, seed=0):
    rng = np.random.default_rng(seed)
    proj = rng.choice(np.eye(in_size), size=out_size, replace=False, axis=1)
    np.save("data/random_proj.npy", proj)

if __name__ == "__main__":
    fire.Fire(main)
