import pandas as pd
from pandas import DataFrame

from formatting import process_and_print, sep


def format_overall_col(col):
    return "{col}_{suffix}".format(
        col=col,
        suffix=SUFFIX_OVERALL
    )


CLASSIFICATION_COL = "classification"
DATA_COL = "text"
COUNT_COL = "count"
CHANCE_COL = "chance"
PERCENT_COL = "%"

SUFFIX_OVERALL = "overall"


OVERALL_CLASSIFICATION_COL = format_overall_col(CLASSIFICATION_COL)
OVERALL_COUNT_COL = format_overall_col(COUNT_COL)
OVERALL_CHANCE_COL = format_overall_col(CHANCE_COL)
OVERALL_PERCENT_COL = format_overall_col(PERCENT_COL)


def read_stop_words(stop_words_path: str) -> list[str]:
    stop_words = open(stop_words_path, "r").readlines()
    stop_words = list(map(lambda x: x.strip().lower(), stop_words))
    return stop_words


def sanitize_and_explode_words(df: DataFrame, stop_words: list[str]):
    df[DATA_COL] = df[DATA_COL].str.split().transform(
        lambda l: list(map(lambda w: w.strip().lower(), l))
    ).transform(
        lambda l: list(filter(lambda w: [w for c in w if c.isalpha()], l))
    )
    words = df.explode(str(DATA_COL))

    words_filtered = words[~words[DATA_COL].isin(stop_words)]

    return words_filtered


def group_count_words(
    words,
    cols=[DATA_COL]
):
    return words.groupby(
        # type: ignore
        cols,
        observed=True,
        sort=False,
    ).size().reset_index(name=COUNT_COL)


def get_num_words_per_classification(df: DataFrame):
    sep()
    print("Getting num words per classification")

    grouping = process_and_print(
        label="Grouping",
        process=lambda: df.groupby(
            [
                CLASSIFICATION_COL,
                DATA_COL
            ],
            observed=True,
            as_index=False
        ).sum().groupby(CLASSIFICATION_COL).count()
    )

    to_records = process_and_print(
        label="To records",
        process=lambda: grouping.T.to_dict(orient='records')[0]
    )

    return to_records
