models = test EleutherAI/pythia-70m

all: paper 

test: %: scripts/pythia.py data/%/coeffs3.pt data/%/weight.pt 
	python $< --model $@

EleutherAI/pythia-70m: EleutherAI/%: scripts/pythia.py data/%/coeffs3.pt data/%/weight.pt 
	python $< --model $@

data/%/coeffs3.pt: scripts/save_coeffs3.py data/%/est_bias.pt data/%/points.pt scripts/utils.py
	python $< --model $*

data/%/est_bias.pt: scripts/save_coeffs2.py data/%/bias.pt data/%/coeffs1.pt scripts/utils.py
	python $< --model $*

data/%/points.pt data/%/coeffs1.pt: scripts/save_coeffs1.py data/%/logits.pt scripts/utils.py
	python $< --model $* --device cpu

data/test/logits.pt data/test/bias.pt: scripts/save_logits_and_params.py
	python $< --model test --device cpu

data/pythia-70m/logits.pt data/pythia-70m/bias.pt: scripts/save_logits_and_params.py
	python $< --model EleutherAI/pythia-70m --device cpu

queries: scripts/make_batch_file.py
	python $<
	
paper: tab/models.tex data/fit.table data/extrapolate.table
	latexmk --pdf exact &> /dev/null; pplatex -i exact.log 

tab/models.tex: scripts/cost_est.py
	python $<

overleaf/data/fit.table overleaf/data/extrapolate.table: overleaf/data/%.table: scripts/%.gnuplot data/times.dat
	gnuplot $<

clean:
	latexmk -C exact
