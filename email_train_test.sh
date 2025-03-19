#!/usr/bin/env bash

python3 naive_bayes.py -s ./stop_words.txt train-test -ad ./tests/30_records/train.csv --split 80
