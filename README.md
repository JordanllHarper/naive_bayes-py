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
    - `sh email_train_test.sh` - Bash script included for convenience in running a test and train on the `emails.csv` data set using an "60% train 40% test" split, `sh email_train_test.sh`, using stop words defined in ./stop_words.txt.

## Usage

The program is split into two modes:
- `train-test` will take your data set, divide it into training data and test data, and give out results for that.
- `manual` requires you to specify your own training data, and then a test email to apply the model to.

### Common arguments

- `-s` `--stopwords` - `.txt` file of the words you want the algorithm to filter out. This isn't required but is highly recommended for more realistic predictions.
- `--codec` - the codec to apply when reading the CSV file. Defaults to `utf-8`.
- `-- bias` - the bias the algorithm will apply to each entry. In practice, this will increase the count of a given entry by this amount. Defaults to 1.
- `--datacolumn` - the column in the CSV input that specifies the "data" of the entry. For instance, in the `emails.csv` file, this column is the "text" column with the spam email contents. Defaults to 0 (the first column).
- `--classcolumn` - the column in the CSV input that specifies the "classification" of the entry. For instance, in the `emails.csv` file, this column is the "spam" column with the email classification. Defaults to 1 (the second column).

### train-test arguments

- `-d` `--data` - the path to the data source that will be split into training and test data.
- `--split` - the split to apply to the data source in CSV format. E.g. give a value of `50` for an 50% split of training and test data. Default is 80 or 80% to 20% split.

### manual arguments

- `-d` `--data` - the path to the training data in CSV format.
- `-t` `--test` - the path to the test data in CSV format.
- `-m` `--model` - either:
    - a path to a pretrained model if the data and stopwords arguments are ommitted.
    - or a path to save a model if the data and stopwords are provided.
> [!NOTE]
> This means you can separate out your training and testing pipeline as required.
