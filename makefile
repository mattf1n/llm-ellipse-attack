all: data/ellipse_pred_5000_samples.npz

data/single_token_prompts/outputs.npz: scripts/sample_from_model.py
	python $<

data/single_token_prompts/logprobs.npy: scripts/logprobs_of_logits.py data/single_token_prompts/outputs.npz
	python $<

data/ellipse_pred_5000_samples.npz: scripts/time_ellipse_solving.py data/single_token_prompts/logprobs.npy
	python $<
