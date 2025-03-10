from pandas import DataFrame
import pandas as pd
from formatting import *
from util import *


def standardize_column_names(df, data_col_index, class_col_index) -> DataFrame:
    data_column, classification_column = str(
        df.columns[int(data_col_index)]), str(df.columns[int(class_col_index)])
    df = df.rename(
        columns={
            classification_column: CLASSIFICATION_COL,
            data_column: DATA_COL
        }
    )
    df[CLASSIFICATION_COL] = df[CLASSIFICATION_COL].astype("category")
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df


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
        process=lambda: group_count_words(
            words,
            cols=[
                CLASSIFICATION_COL,
                DATA_COL
            ]
        ),
    )

    num_words_per_classification = process_and_print(
        label="Number of words per classification",
        process=lambda: get_num_words_per_classification(
            words_grouped,
        )
    )

    words_grouped["mapped"] = process_and_print(
        label="Mapping classifications",
        process=lambda: words_grouped[CLASSIFICATION_COL].map(
            lambda x: num_words_per_classification[0][x]
        )
    )

    print(words_grouped)

    words_grouped[CHANCE_COL] = process_and_print(
        label="Chance calculated",
        process=lambda:
            words_grouped[COUNT_COL] / words_grouped["mapped"].astype(int)
    )
    print(words_grouped)

    words_grouped["%"] = process_and_print(
        label="% chance",
        process=lambda: words_grouped[CHANCE_COL] * 100
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


def train_model(
        path_to_csv: str,
        stop_words: list[str],
        data_col_index,
        class_col_index,
        codec,
) -> DataFrame:
    df = pd.read_csv(path_to_csv, encoding=codec)

    sep()
    print_with_header("Data")
    print(df)
    print(stop_words)

# ---

    sep()

# ---

    df = standardize_column_names(
        df,
        data_col_index,
        class_col_index
    )
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

    step_print(2, "Map words and their classification counts")
    model = map_words_to_classification_counts(
        df,
        stop_words,
    )

    step_print(3, "Merge words and classification probabilities")
    model = model.join(
        df_overall_classification_chance,
        rsuffix="_{col}".format(col=SUFFIX_OVERALL),
        on=CLASSIFICATION_COL,
    )

    print(model)

    return model
