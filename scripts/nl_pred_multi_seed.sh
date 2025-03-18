#!/bin/bash

for seed in $(seq 10)
do 
	for sample_size in 5000 10000 20000 50000
	do 
		outfile=data/pile-uncopyrighted/ellipse_pred/seeds/${seed}_seed_${sample_size}_samples.npz
		mkdir -p $(dirname $outfile)
		python scripts/time_ellipse_solving.py \
			--infile data/pile-uncopyrighted/TinyStories-1M/outputs.npz  \
			--outfile=$outfile \
			--sample-size=$sample_size \
			--down-proj=data/random_proj.npy \
			--random \
			--seed=$seed
	done
done
