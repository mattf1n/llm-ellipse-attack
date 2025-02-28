all: data/ellipse_pred_5000_samples.npz

data/single_token_prompts/outputs.npz: scripts/sample_from_model.py
	python $<

data/single_token_prompts/logprobs.npy: scripts/logprobs_of_logits.py data/single_token_prompts/outputs.npz
	python $< < data/single_token_prompts/outputs.npz > $@

data/ellipse_pred_5000_samples.npz: scripts/time_ellipse_solving.py data/single_token_prompts/logprobs.npy
	python $< --sample-size=5000

data/single_token_prompts/narrow_band_logits.npy: scripts/save_hidden_size_centered_norms.py data/single_token_prompts/outputs.npz
	python $<

data/single_token_prompts/narrow_band_logprobs.npy: scripts/logprobs_of_logits.py data/single_token_prompts/narrow_band_logits.npz
	python $< < data/single_token_prompts/narrow_band_logits.npz > $@

data/narrow_band_ellipse_pred_5000_samples.npz: scripts/time_ellipse_solving.py data/single_token_prompts/narrow_band_logprobs.npy
	python $< --filter --sample-size=5000
