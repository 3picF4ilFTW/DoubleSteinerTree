#!/bin/bash

for f in *.txt
do
	python main.py $f 0.5 900 > /dev/null &
done
