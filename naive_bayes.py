import argparse

from formatting import print_with_header, sep, space, step_print
from train import train_model
from test import test_model
import pandas as pd


from util import read_stop_words


def present(result):
    print("Classification : Chance : %")
    space()
    for k, v in result.items():
        print(f"{k} : {v} : {v * 100}")
        space()


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

    parser.add_argument(
        "-dc",
        "--datacolumn",
        default=0,
        help="Specify the column index in the training CSV to treat as data. Defaults to 0 (the first column)"
    )
    parser.add_argument(
        "-cc",
        "--classcolumn",
        default=1,
        help="Specify the column index in the training CSV to treat as classification. Defaults to 1 (the second column)"
    )

    parser.add_argument(
        "-co",
        "--codec",
        default="utf-8",
        help="Specify the data codec form. Defaults to UTF-8."
    )

    args = parser.parse_args()

    step_print(0, "Config and Setup")
    path_to_training_data, stop_words_path, test_data, model_path, data_col_index, class_col_index, codec = args.data, args.stopwords, args.test, args.model, args.datacolumn, args.classcolumn, args.codec

    print("Training CSV path:", path_to_training_data)
    print("Stop words path:", stop_words_path)
    print("Test data path:", test_data)
    print("Configured model", model_path)

    print("Specified data column:", data_col_index)
    print("Specified classification column:", class_col_index)
    print("Specified codec:", codec)

    model = None

    should_train = path_to_training_data != None
    stop_words = \
        read_stop_words(stop_words_path) if stop_words_path != None else []

    if should_train:
        sep()
        print_with_header("Training model")
        model = train_model(
            path_to_training_data,
            stop_words,
            data_col_index,
            class_col_index,
            codec,
        )
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
        sep()
        print("Results")
        sep()
        present(result)
