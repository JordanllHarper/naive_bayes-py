import pandas as pd
import step_1
import step_2
from formatting import *


def read_stop_words(stop_words_path: str | None) -> list[str]:
    if stop_words_path == None:
        return []
    else:
        stop_words = open(stop_words_path, "r").readlines()
        stop_words = list(map(lambda x: x.strip().lower(), stop_words))
        return stop_words


def train_model(path_to_csv: str, stop_words_path: str | None):
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

    print_with_header("Working out overall classifications")
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
    model: pd.DataFrame = step_2.map_words_to_classification_counts(
        df,
        data_column,
        stop_words,
        classification_column,
    )

    space()

    return model
