import pandas as pd
from typing import Callable
import argparse

import step_1
import step_2


def space():
    print()


def sep():
    print("--------")


def print_with_header(label):
    print("Task:", label)


def process_and_print(
        label: str,
        process: Callable
):
    print_with_header(label)
    result = process()
    print(result)
    space()
    return result


def read_stop_words(stop_words_path) -> list[str]:
    stop_words = open(stop_words_path, "r").readlines()
    stop_words = list(map(lambda x: x.strip().lower(), stop_words))
    return stop_words


def step_print(
    step_num: int,
    message: str
):
    sep()
    print("[STEP {step_num}]: {msg}\n".format(step_num=step_num, msg=message))


if __name__ == "__main__":
    step_print(0, "Config and Setup")
    parser = argparse.ArgumentParser(
        prog="Naive Bayes",
        description="Naive Bayes algorithm implemented in Python using Pandas",
    )

    parser.add_argument("-d", "--data")
    parser.add_argument("-t", "--test")
    parser.add_argument("-s", "--stopwords")
    parser.add_argument("-c", "--configured")

    args = parser.parse_args()
    path_to_csv, stop_words_path, test_data = args.data, args.stopwords, args.test

    print("CSV path:", path_to_csv)
    print("Stop words path:", stop_words_path)
    print("Test data path:", test_data)

    df = pd.read_csv(path_to_csv)
    stop_words = read_stop_words(stop_words_path)

    sep()
    print_with_header("Data")
    print(df)
    print(stop_words)

# ---

    sep()

# ---

    data_column, classification_column = str(df.columns[0]), str(df.columns[1])
    df[classification_column] = df[classification_column].astype("category")
    classification_categories = df[classification_column].unique()

    print("Data column:", data_column)
    print("Classification column:", classification_column)
    print("Classification categories:", classification_categories)

    num_records = df[data_column].count()

    step_print(1, "Overall classifications")
    df_overall_classification_count = step_1.get_classification_counts(
        df,
        classification_column
    )
    df_overall_classification_chance: pd.DataFrame = step_1.get_overall_classification_chance(
        df_overall_classification_count,
        num_records,
    )
    print(df_overall_classification_chance)

    space()

    step_print(2, "Map words and their classification counts")
    step_2.map_words_to_classification_counts(
        df,
        data_column,
        stop_words,
        classification_column,
    )

    space()
