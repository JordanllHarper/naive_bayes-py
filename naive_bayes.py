import argparse

from formatting import print_with_header, sep, space, step_print
from train import train_model
from test import test_model
import pandas as pd


from util import read_stop_words


def present(result):
    print("Classification : Chance to 2 dp : %")
    space()
    for k, v in result.items():
        print(f"{k} : {v} : {v * 100:.2f}")
        space()


def setup_train_test_subcommand(train_test_invocation):
    train_test_invocation.add_argument(
        "-d", "--data", help="path to training and test data in CSV"
    )
    train_test_invocation.add_argument(
        "-sp", "--split", help="the split to use on the dataset"
    )

    return train_test_invocation


def setup_manual_subcommand(manual_invocation):
    manual_invocation.add_argument(
        "-d", "--data", help="path to training data in CSV")
    manual_invocation.add_argument(
        "-t", "--test", help="data to test on in CSV")
    manual_invocation.add_argument(
        "-s", "--stopwords",
        help="the stopwords to filter out of each email"
    )
    manual_invocation.add_argument(
        "-m", "--model",
        help="either a path to a pretrained model if the data and stopwords arguments are ommitted, or a path to save a model if the data and stopwords are provided"
    )
    return manual_invocation


def handle_train_test(args):
    print(args)

    path_to_training_and_test, stop_words_path,  data_col_index, class_col_index, codec, bias = args.data, args.stopwords,  args.datacolumn, args.classcolumn, args.codec, args.bias

    print("Training and test data path", path_to_training_and_test)
    print("Stop words", stop_words_path)
    print("Specified data column:", data_col_index)
    print("Specified classification column:", class_col_index)
    print("Specified codec:", codec)
    print("Specified bias:", bias)

    # TODO: Split data and write

    # model = train_model()

    pass


def handle_manual(args):
    step_print(0, "Config and Setup")
    path_to_training_data, stop_words_path, test_data, model_path, data_col_index, class_col_index, codec, bias = args.data, args.stopwords, args.test, args.model, args.datacolumn, args.classcolumn, args.codec, args.bias

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

        df = pd.read_csv(path_to_training_data, encoding=codec)

        model = train_model(
            df,
            stop_words,
            data_col_index,
            class_col_index,
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
        result = test_model(model, test, stop_words, bias=bias)
        sep()
        print("Results")
        sep()
        present(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Naive Bayes",
        description="""Naive Bayes algorithm implemented in Python using Pandas""",
    )
    parser.add_argument(
        "-s", "--stopwords",
        help="the stopwords to filter out of each email"
    )
    parser.add_argument(
        "-co",
        "--codec",
        default="utf-8",
        help="Specify the data codec form. Defaults to UTF-8."
    )
    parser.add_argument(
        "-b",
        "--bias",
        default=1,
        help="The bias to apply. This will typically be a count of a given entry in the dataset when training.",
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

    subparsers = parser.add_subparsers()

    train_and_test_parser = subparsers.add_parser("train-test")
    manual_parser = subparsers.add_parser("manual")

    train_and_test_subcommand = setup_train_test_subcommand(
        train_and_test_parser
    )
    train_and_test_subcommand.set_defaults(func=handle_train_test)

    manual_subcommand = setup_manual_subcommand(manual_parser)
    manual_subcommand.set_defaults(func=handle_manual)

    args = parser.parse_args()
    try:
        args.func(args)
        print(args)
    except AttributeError as e:
        print("Error! It looks like you tried to invoke this command without any subcommand. See --help for available options")
        print(e)
