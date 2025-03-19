
import numpy as np
from pandas import DataFrame
import pandas as pd

from formatting import process_and_print, sep
from util import *


def test_data_entry(
    for_classification,
    df_model_for_classification: DataFrame,
    entry: str,
    stop_words: list[str],
    bias: int,
):
    sep()

    print(df_model_for_classification)

    classification_chance = process_and_print(
        label="Get overall classification chance",
        process=lambda:
        df_model_for_classification[
            [
                OVERALL_CLASSIFICATION_COL,
                OVERALL_CHANCE_COL
            ]
        ].set_index(OVERALL_CLASSIFICATION_COL).T.to_dict(orient='records')[0][for_classification]
    )

    df_data_exploded = process_and_print(
        label="Sanitize and explode words",
        process=lambda: sanitize_and_explode_words(
            pd.DataFrame({
                DATA_COL: [entry],
            }),
            stop_words
        )
    )

    sep()

    df_data_words = process_and_print(
        label="Count data words",
        process=lambda: group_count_words(
            df_data_exploded,
            cols=[DATA_COL]
        )
    )

    df_data_words[COUNT_COL] = df_data_words[COUNT_COL] + float(bias)

    result = process_and_print(
        label="Merge",
        process=lambda: pd.merge(
            left=df_model_for_classification,
            right=df_data_words,
            on=DATA_COL,
            suffixes=("_model", "_data")
        )
    )

    # Log implementation informed by (Rhodes College, n.d.) to prevent extremely small numbers. Still compare as before.
    result[CHANCE_COL] = np.log(result[CHANCE_COL])

    final = process_and_print(
        label="Final result for classification",
        process=lambda: result[CHANCE_COL].sum() * classification_chance
    )

    return final


def make_assertion(row):
    row = row.filter(regex="[^text, ^classification]")
    dict_row = row.to_dict()
    return max(dict_row, key=dict_row.get)


def test_model(
    df_model: DataFrame,
    df_data: DataFrame,
    stop_words: list[str],
    bias: int,
) -> DataFrame:
    col = str(df_data.columns[0])
    df_data.rename(columns={col: DATA_COL})
    classifications = df_model[CLASSIFICATION_COL].unique()
    for classification in classifications:
        df_model_for_classification = df_model[
            df_model[CLASSIFICATION_COL] == classification
        ]
        df_data[f"{classification}"] = df_data[DATA_COL].apply(
            lambda x: test_data_entry(
                for_classification=classification,
                df_model_for_classification=df_model_for_classification,
                entry=x,
                stop_words=stop_words,
                bias=bias
            )
        )

    df_data["classification_assertion"] = df_data.apply(
        lambda row: make_assertion(row),
        raw=False,
        axis=1,
    )
    return df_data
