models = test EleutherAI/pythia-70m

all: paper $(models)

test: %: scripts/pythia.py data/%/coeffs3.pt data/%/weight.pt 
	python $< --model $@

EleutherAI/pythia-70m: EleutherAI/%: scripts/pythia.py data/%/coeffs3.pt data/%/weight.pt 
	python $< --model $@

data/%/coeffs3.pt: scripts/save_coeffs3.py data/%/est_bias.pt data/%/points.pt
	python $< --model $*

data/%/est_bias.pt: scripts/save_coeffs2.py data/%/bias.pt data/%/coeffs1.pt
	python $< --model $*

data/%/points.pt data/%/coeffs1.pt: scripts/save_coeffs1.py data/%/logits.pt
	python $< --model $*

data/test/logits.pt data/test/bias.pt: scripts/save_logits_and_params.py
	python $< --model test

data/pythia-70m/logits.pt data/pythia-70m/bias.pt: scripts/save_logits_and_params.py
	python $< --model EleutherAI/pythia-70m
	
paper: 
	latexmk --pdf exact &> /dev/null; pplatex -i exact.log 

clean:
	latexmk -C exact
