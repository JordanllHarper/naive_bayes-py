import argparse

from formatting import print_with_header, step_print
from train import train_model
from test import test_model
from present import present_results
import pandas as pd

from util import read_stop_words


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Naive Bayes",
        description="""Naive Bayes algorithm implemented in Python using Pandas""",
    )

    parser.add_argument("-d", "--data", help="path to training data in CSV")
    parser.add_argument("-t", "--test", help="data to test on in CSV")
    parser.add_argument(
        "-s", "--stopwords",
        help="the stopwords to filter out of each email"
    )
    parser.add_argument(
        "-m", "--model",
        help="either a path to a pretrained model if the data and stopwords arguments are ommitted, or a path to save a model if the data and stopwords are provided"
    )

    args = parser.parse_args()

    step_print(0, "Config and Setup")
    path_to_training_data, stop_words_path, test_data, model_path = args.data, args.stopwords, args.test, args.model

    print("CSV path:", path_to_training_data)
    print("Stop words path:", stop_words_path)
    print("Test data path:", test_data)
    print("Configured model", test_data)

    model = None

    should_train = path_to_training_data != None
    stop_words = \
        read_stop_words(stop_words_path) if stop_words_path != None else []

    if should_train:
        print_with_header("Training model")
        model = train_model(path_to_training_data, stop_words)
        if model_path != None:
            print_with_header(
                "Exporting to CSV at {path}".format(path=model_path)
            )
            model.to_csv(model_path)
    # model path is supplied already, read from CSV
    elif model_path != None:
        print_with_header("Importing model")
        model = pd.read_csv(model_path)
    # we need at least one option here
    else:
        raise Exception(
            "Expected either a path to training data or a path to a pretrained model to import, both in CSV"
        )
    if test_data != None:
        print_with_header("Testing using provided model")
        test = pd.read_csv(test_data)
        result = test_model(model, test, stop_words)
        print_with_header("Results")
        present_results(result)
