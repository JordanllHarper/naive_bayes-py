import pandas as pd

import sys

import step_1
import step_2


def sep():
    print()


def read_stop_words(stop_words_path) -> list[str]:
    stop_words = open(stop_words_path, "r").readlines()
    stop_words = list(map(lambda x: x.strip().lower(), stop_words))
    return stop_words


def step_print(step_num: int, message: str):
    print("--------")
    print("[STEP {step_num}]: {msg}\n".format(step_num=step_num, msg=message))


if __name__ == "__main__":
    step_print(0, "Setup")
    args = sys.argv[1:]
    path_to_csv, stop_words_path, test_data = args[0], args[1], args[2]

    print("CSV path:", path_to_csv)
    print("Stop words path:", stop_words_path)
    print("Test data path:", test_data)

    df = pd.read_csv(path_to_csv)
    stop_words = read_stop_words(stop_words_path)

    print(df)
    print(stop_words)

# ---

    sep()

# ---

    data_column, classification_column = str(df.columns[0]), str(df.columns[1])
    df[classification_column] = df[classification_column].astype("category")
    classification_categories = df[classification_column].unique()

    print("Data header:", data_column)
    print("Classification header:", classification_column)
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

    sep()

    step_print(2, "Map words and their classification counts")
    step_2.map_words_to_classification_counts(
        df,
        data_column,
        stop_words,
        classification_column,
    )

    sep()
