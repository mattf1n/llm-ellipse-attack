all: data/error.txt data/pile-uncopyrighted/TinyStories-3M/outputs.npz

clean:
	rm data/ellipse_pred_5000_samples.npz \
		data/ellipse_pred_10000_samples.npz \
		data/ellipse_pred_5000_samples_random_proj.npz \
		data/narrow_band_ellipse_pred_5000_samples.npz

# Sampling from the model

data/single_token_prompts/outputs.npz: 
	python scripts/sample_from_model.py

data/pile-uncopyrighted/TinyStories-1M/outputs.npz: 
	python scripts/sample_from_model.py --dataset=monology/pile-uncopyrighted --batch-size=100 --samples=500_000

data/pile-uncopyrighted/TinyStories-3M/outputs.npz:
	python scripts/sample_from_model.py --dataset=monology/pile-uncopyrighted --batch-size=10 --model-name=roneneldan/TinyStories-3M


# Cannonical down-projection (first emb) varying sample size, randomizing samples.

sample_sizes=5000 10000 20000 50257
pred_file_names_=$(addprefix data/ellipse_pred_, $(sample_sizes))
pred_file_names=$(addsuffix _samples.npz, $(pred_file_names_))

$(pred_file_names): data/ellipse_pred_%_samples.npz: \
	data/single_token_prompts/logprobs.npy
	python scripts/time_ellipse_solving.py \
		--infile=$< \
		--outfile=$@ \
		--sample-size=$*

random_sample_pred_fnames=$(addsuffix _random_sample.npz, $(basename $(pred_file_names)))

$(random_sample_pred_fnames): data/ellipse_pred_%_samples_random_sample.npz: \
	data/single_token_prompts/logprobs.npy
	python scripts/time_ellipse_solving.py \
		--infile=$< \
		--outfile=$@ \
		--sample-size=$* \
		--random \
		--seed=0


nl_sample_sizes = $(sample_sizes) 100000
nl_pred_fnames_ = $(addprefix data/pile-uncopyrighted/ellipse_pred/, $(nl_sample_sizes))
nl_pred_fnames = $(addsuffix _samples.npz, $(nl_pred_fnames_))

$(nl_pred_fnames): data/pile-uncopyrighted/ellipse_pred/%_samples.npz: \
	data/pile-uncopyrighted/TinyStories-1M/outputs.npz
	mkdir -p $(dir $@)
	python scripts/time_ellipse_solving.py \
		--infile=$< \
		--outfile=$@ \
		--sample-size=$* \
		--down-proj=data/random_proj.npy



clean_nl: 
	rm $(nl_pred_fnames)
		



# Generating a random projection

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

narrow_band_pred_fnames = data/narrow_band_ellipse_pred_5000_samples.npz \
			  data/narrow_band_ellipse_pred_10000_samples.npz

$(narrow_band_pred_fnames): data/narrow_band_ellipse_pred_%_samples.npz: \
	data/single_token_prompts/narrow_band_logprobs.npy
	python scripts/time_ellipse_solving.py \
		--infile=$< \
		--outfile=$@ \
		--narrow-band --sample-size=$*

clean_narrow_band:
	rm $(narrow_band_pred_fnames)

data/wide_band_ellipse_pred_5000_samples.npz: \
	data/single_token_prompts/wide_band_logprobs.npy
	python scripts/time_ellipse_solving.py \
		--outfile=$@ \
		--infile=$< \
		--wide-band --sample-size=5000

random_proj_pred_fnames = data/ellipse_pred_5000_samples_random_proj.npz \
			  data/ellipse_pred_20000_samples_random_proj.npz

$(random_proj_pred_fnames): data/ellipse_pred_%_samples_random_proj.npz: \
	data/single_token_prompts/narrow_band_logprobs.npy \
	data/random_proj.npy
	python scripts/time_ellipse_solving.py \
		--infile=$< \
		--outfile=$@ \
		--narrow-band \
		--sample-size=$* \
		--down-proj=data/random_proj.npy

data/error_data.pkl: \
	scripts/validate_ellipse_pred.py \
	data/ellipse_pred_5000_samples.npz \
	data/ellipse_pred_10000_samples.npz \
	data/ellipse_pred_20000_samples.npz \
	data/ellipse_pred_50257_samples.npz \
	$(random_proj_pred_fnames) \
	$(narrow_band_pred_fnames) \
	data/wide_band_ellipse_pred_5000_samples.npz \
	data/ellipse_pred_5000_samples_random_sample.npz \
	data/ellipse_pred_10000_samples_random_sample.npz \
	data/ellipse_pred_20000_samples_random_sample.npz \
	data/ellipse_pred_50257_samples_random_sample.npz \
	$(nl_pred_fnames)
	python $^

data/error.txt: data/error_data.pkl
	python -c "import pandas as pd; pd.read_pickle('data/error_data.pkl').T.to_string('data/error.txt')"
