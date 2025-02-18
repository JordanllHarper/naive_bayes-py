from pandas import DataFrame
import pandas as pd
from formatting import *
from util import *


def get_classification_counts(df: DataFrame):
    return DataFrame(
        {
            CLASSIFICATION_COL_FMT: df[CLASSIFICATION_COL].unique(),
            "count": df[CLASSIFICATION_COL].value_counts()
        }
    )


def get_overall_classification_chance(df_overall_classification_counts: DataFrame, num_records):
    df_overall_classification_chance = df_overall_classification_counts.copy()

    df_overall_classification_chance[CHANCE_COL_FMT] = df_overall_classification_chance["count"] / num_records
    df_overall_classification_chance[PERCENT_COL_FMT] = df_overall_classification_chance[CHANCE_COL_FMT] * 100

    return df_overall_classification_chance


def get_num_words_per_classification(df):
    return df.groupby(
        CLASSIFICATION_COL,
        observed=True
    ).size().reset_index(name="count").set_index(CLASSIFICATION_COL).T.to_dict(orient='records')[0]


def map_words_to_classification_counts(
        start_df: DataFrame,
        stop_words: list,
):

    def sanitize_and_explode_words(df: DataFrame):
        df[DATA_COL] = df[DATA_COL].str.split().transform(
            lambda l: list(map(lambda w: w.strip().lower(), l))
        ).transform(
            lambda l: list(filter(lambda w: [w for c in w if c.isalpha()], l))
        )
        df = df.explode(str(DATA_COL))

        return df

    words = sanitize_and_explode_words(start_df)
    print(words)
    space()

    # Dataframe structure:
    # Word | Classification 1 Count | Classification 2 Count | ...

    words_filtered = process_and_print(
        label="Words filtered",
        process=lambda:
            words[~words[DATA_COL].isin(stop_words)]
    )

    words_grouped = process_and_print(
        label="Word groups",
        process=lambda: words_filtered.groupby(
            # type: ignore
            [
                CLASSIFICATION_COL,
                DATA_COL
            ],
            observed=True,
            sort=False,
        ).size().reset_index(name="count")
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

    data_column, classification_column = str(df.columns[0]), str(df.columns[1])
    df = df.rename(
        columns={classification_column: CLASSIFICATION_COL, data_column: DATA_COL})
    df[CLASSIFICATION_COL] = df[CLASSIFICATION_COL].astype("category")
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
