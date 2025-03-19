#!/usr/bin/env bash

python3 naive_bayes.py -s ./stop_words.txt manual -d ./tests/small/train.csv -t ./tests/small/test.csv
