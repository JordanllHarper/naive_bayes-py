# Multinomial Naive Bayes Classification Algorithm

An implementation of the Multinomial Naive Bayes Classification Algorithm in Python using the pandas library.

## Requirements

- [Python](https://www.python.org/downloads/)
- [Pandas](https://pandas.pydata.org/docs/getting_started/install.html)

## Getting started

1. Install [Requirements](#requirements)
2. Clone the repository `git clone https://github.com/JordanllHarper/naive_bayes-py`
3. Run the program
    - `python3 naive_bayes.py --help` for command line help.
    - `sh email_train_test.sh` - Bash script included for convenience in running a test and train on the `emails.csv` data set using an "80% train 20% test" split using stop words defined in stop_words.txt.

## Usage

The program is split into two modes:

- `train-test` will take your data set, divide it into training data and test data, and give out results.
- `manual` requires you to specify your own training data and test data.
    
> [!IMPORTANT]
>  When using manual mode, your training and test data MUST have the same schema. e.g. if you have "text" and "spam" fields in training, these also need to be present in testing.

### Common arguments

- `-s` `--stopwords` (string) - path to a `.txt` file of the words you want the algorithm to filter out. This isn't required but is highly recommended for more realistic predictions.
- `--codec` (string) - the codec to apply when reading the CSV file. Defaults to `utf-8`.
- `--bias` (float) - the bias the algorithm will apply to each entry. In practice, this will increase the count of a given entry by this amount. Defaults to `1.`
- `--datacolumn` (int) - the column in the CSV input that specifies the "data" of the entry. For instance, in the `emails.csv` file, this column is the "text" column with the spam email contents. Defaults to `0` (the first column).
- `--classcolumn` (int) - the column in the CSV input that specifies the "classification" of the entry. For instance, in the `emails.csv` file, this column is the "spam" column with the email classification. Defaults to `1` (the second column).
- `--export` (string) - Export path for test results. Defaults to None (no export).


### train-test arguments

- `-d` `--data` (string) - the path to the data source that will be split into training and test data.
- `--split` (float) - the split to apply to the data source in CSV format. E.g. give a value of `50` for an 50% split of training and test data. Default is `80` (80% training to 20% test split).

### manual arguments

- `-d` `--data` - the path to the training data in CSV format.
- `-t` `--test` - the path to the test data in CSV format.
- `-m` `--model` - either:
    - a path to a pre-trained model if the data and stopwords arguments are ommitted.
    - or a path to save a model if the data and stopwords are provided.

## Examples

"Use the manual mode which trains on the `emails.csv` file and tests on the `test_data.csv` without saving the model. Add a bias of 3 and use the stopwords `stop_words.txt`":

`python3 naive_bayes.py --bias 3.0 --stopwords stop_words.txt manual -d emails.csv -t test_data.csv`

"Use the train-test mode on a `more_emails.csv` file, specifying the data column as 1 and classification column as 0":

`python3 naive_bayes.py --stopwords more_stopwords.txt --datacolumn 1 --classcolumn 0 train-test -d more_emails.csv`

> [!NOTE]
> The algorithm gives very small numbers, so a log function is used to balance this out. Compare the 2 negative numbers and take the closest to 0 for the classification, it should work the same.
