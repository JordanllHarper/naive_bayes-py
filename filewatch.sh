#!/usr/bin/env bash

while true; do
    fswatch -o . | python3 naive_bayes.py -d emails.csv -s stop_words.txt -t test_data.csv
done
