#!/bin/bash

for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
	echo "running all instances with alpha=${alpha}"
	for f in *.txt
	do
		python main.py $f $alpha 900 > /dev/null &
	done
	wait
	echo "moving files into results_${alpha}"
	mkdir results_$alpha
	mv *_vnd.txt results_$alpha
	mv *_grasp.txt results_$alpha
	mv *_log.txt results_$alpha
done
