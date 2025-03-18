#!/usr/bin/env bash

python3 naive_bayes.py -s ./stop_words.txt train-test  -d ./emails.csv --split 80
