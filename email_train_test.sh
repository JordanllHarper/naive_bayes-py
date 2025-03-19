#!/usr/bin/env bash

python3 naive_bayes.py -s ./stop_words.txt train-test -d ./tests/30_records/train.csv --split 80
