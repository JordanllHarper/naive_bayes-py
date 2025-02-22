from pandas import DataFrame
import pandas as pd
from formatting import *
from util import *


def get_classification_counts(df: DataFrame):
    return DataFrame(
        {
            OVERALL_CLASSIFICATION_COL: df[CLASSIFICATION_COL].unique(),
            OVERALL_COUNT_COL: df[CLASSIFICATION_COL].value_counts()
        }
    )


def get_overall_classification_chance(df_overall_classification_counts: DataFrame, num_records):
    df_overall_classification_chance = df_overall_classification_counts.copy()

    df_overall_classification_chance[OVERALL_CHANCE_COL] = df_overall_classification_chance[OVERALL_COUNT_COL] / num_records
    df_overall_classification_chance[OVERALL_PERCENT_COL] = df_overall_classification_chance[OVERALL_CHANCE_COL] * 100

    return df_overall_classification_chance


def map_words_to_classification_counts(
        start_df: DataFrame,
        stop_words: list[str],
):

    words = sanitize_and_explode_words(start_df, stop_words)
    print(words)
    space()

    # Dataframe structure:
    # Word | Classification 1 Count | Classification 2 Count | ...

    words_grouped = process_and_print(
        label="Word groups",
        process=lambda: group_words(words),
    )

    num_words_per_classification = process_and_print(
        label="Number of words per classification",
        process=lambda: get_num_words_per_classification(
            words_grouped,
        )
    )

    words_grouped["chance"] = process_and_print(
        label="Chance calculated",
        process=lambda: words_grouped["count"] /
        words_grouped[CLASSIFICATION_COL].map(
            num_words_per_classification
        ).astype(int)
    )

    words_grouped["%"] = process_and_print(
        label="% chance",
        process=lambda: words_grouped["chance"] * 100
    )

    process_and_print(
        label="Sorted by %",
        process=lambda:
            words_grouped.sort_values(
                "%",
                ascending=False,
            )
    )

    return words_grouped


def train_model(path_to_csv: str, stop_words: list[str]) -> DataFrame:
    df = pd.read_csv(path_to_csv)

    sep()
    print_with_header("Data")
    print(df)
    print(stop_words)

# ---

    sep()

# ---

    df = standardize_column_names(df)
    classification_categories = df[CLASSIFICATION_COL].unique()

    print("Classification categories:", classification_categories)

    num_records = df[CLASSIFICATION_COL].count()

    print_with_header("Working out overall classifications")
    df_overall_classification_count = get_classification_counts(
        df,
    )
    df_overall_classification_chance: DataFrame = get_overall_classification_chance(
        df_overall_classification_count,
        num_records,
    )
    print(df_overall_classification_chance)

    space()

    step_print(2, "Map words and their classification counts")
    model = map_words_to_classification_counts(
        df,
        stop_words,
    )

    space()

    step_print(3, "Merge words and classification probabilities")
    model = model.join(
        df_overall_classification_chance,
        rsuffix="_{col}".format(col=SUFFIX_OVERALL)
    )

    space()

    print(model)

    return model
