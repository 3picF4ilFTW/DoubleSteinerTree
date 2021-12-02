#!/bin/bash

for alpha_off in -75 -50 -25 25 50 75
do
	echo "running instances with alpha_off=${alpha_off}"
	alpha=$((700 + alpha_off))
	python main.py 0049.txt 0.$alpha 900 > /dev/null &
	alpha=$((600 + alpha_off))
	python main.py 0125.txt 0.$alpha 900 > /dev/null &
	alpha=$((600 + alpha_off))
	python main.py 1331.txt 0.$alpha 900 > /dev/null &
	wait
	echo "moving files into results_${alpha_off}"
	mkdir results_$alpha_off
	mv *_vnd.txt results_$alpha_off
	mv *_grasp.txt results_$alpha_off
	mv *_log.txt results_$alpha_off
done
