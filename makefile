all: data/error_data.pkl

clean:
	rm data/ellipse_pred_5000_samples.npz \
		data/ellipse_pred_10000_samples.npz \
		data/ellipse_pred_5000_samples_random_proj.npz \
		data/narrow_band_ellipse_pred_5000_samples.npz

data/single_token_prompts/outputs.npz: scripts/sample_from_model.py
	python $<

data/single_token_prompts/logprobs.npy: \
	scripts/logprobs_of_logits.py \
	data/single_token_prompts/outputs.npz
	python $< < data/single_token_prompts/outputs.npz > $@

data/ellipse_pred_5000_samples.npz: \
	data/single_token_prompts/logprobs.npy
	python scripts/time_ellipse_solving.py \
		--infile=$< \
		--outfile=$@ \
		--sample-size=5000

data/ellipse_pred_10000_samples.npz: \
	data/single_token_prompts/logprobs.npy
	python scripts/time_ellipse_solving.py \
		--infile=$< \
		--outfile=$@ \
		--sample-size=10000

data/ellipse_pred_20000_samples.npz: \
	data/single_token_prompts/logprobs.npy
	python scripts/time_ellipse_solving.py \
		--infile=$< \
		--outfile=$@ \
		--sample-size=20000

data/random_proj.npy: \
	scripts/generate_random_proj.py
	python $< --in-size=50256 --out-size=63

data/single_token_prompts/narrow_band_logits.npz data/single_token_prompts/wide_band_logits.npz: \
	scripts/save_hidden_size_centered_norms.py \
       	data/single_token_prompts/outputs.npz
	python $<

data/single_token_prompts/narrow_band_logprobs.npy: \
	scripts/logprobs_of_logits.py \
	data/single_token_prompts/narrow_band_logits.npz
	python $< < data/single_token_prompts/narrow_band_logits.npz > $@

data/single_token_prompts/wide_band_logprobs.npy: \
	scripts/logprobs_of_logits.py \
	data/single_token_prompts/wide_band_logits.npz
	python $< < data/single_token_prompts/wide_band_logits.npz > $@

data/narrow_band_ellipse_pred_5000_samples.npz: \
	data/single_token_prompts/narrow_band_logprobs.npy
	python scripts/time_ellipse_solving.py \
		--infile=$< \
		--outfile=$@ \
		--narrow-band --sample-size=5000

data/wide_band_ellipse_pred_5000_samples.npz: \
	data/single_token_prompts/wide_band_logprobs.npy
	python scripts/time_ellipse_solving.py \
		--outfile=$@ \
		--infile=$< \
		--wide-band --sample-size=5000

data/narrow_band_ellipse_pred_5000_samples_random_proj.npz: \
	data/single_token_prompts/narrow_band_logprobs.npy \
	data/random_proj.npy
	python scripts/time_ellipse_solving.py \
		--infile=$< \
		--outfile=$@ \
		--narrow-band \
		--sample-size=5000 \
		--down-proj=data/random_proj.npy

data/ellipse_pred_5000_samples_random_proj.npz: \
	data/single_token_prompts/logprobs.npy \
	data/random_proj.npy
	python scripts/time_ellipse_solving.py \
		--infile=$< \
		--outfile=$@ \
		--sample-size=5000 \
		--down-proj=data/random_proj.npy


data/error_data.pkl: \
	scripts/validate_ellipse_pred.py \
	data/ellipse_pred_5000_samples.npz \
	data/ellipse_pred_10000_samples.npz \
	data/ellipse_pred_20000_samples.npz \
	data/ellipse_pred_5000_samples_random_proj.npz \
	data/narrow_band_ellipse_pred_5000_samples.npz \
	data/wide_band_ellipse_pred_5000_samples.npz
	python $^
