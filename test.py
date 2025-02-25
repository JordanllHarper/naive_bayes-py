
from pandas import DataFrame
import pandas as pd

from formatting import process_and_print, sep
from util import *


def calculate_chance_for_classification(
    classification,
    classification_chance,
    df_model,
    df_data
):
    df_model_for_classification = df_model[
        df_model[CLASSIFICATION_COL] == classification
    ]

    result = pd.merge(
        left=df_model_for_classification,
        right=df_data,
        on=DATA_COL,
        suffixes=("_model", "_data")
    )

    MULTIPLE_COL = "multiplied"

    result[MULTIPLE_COL] = result[CHANCE_COL] * result["count_data"]
    summed = result[MULTIPLE_COL].prod() * classification_chance
    return summed


def test_model(
    df_model: DataFrame,
    df_data: DataFrame,
    stop_words: list[str]
) -> dict:
    col = str(df_data.columns[0])
    df_data = df_data.rename(columns={col: DATA_COL})
    df_model_words = df_model[
        [
            CLASSIFICATION_COL,
            DATA_COL,
            COUNT_COL,
            CHANCE_COL,
        ]
    ]

    df_model_classifications = df_model[CLASSIFICATION_COL].unique()
    sep()

    print(df_model)

    classification_chance = process_and_print(
        label="Get overall classification chance",
        process=lambda:
        df_model[
            [
                OVERALL_CLASSIFICATION_COL,
                OVERALL_CHANCE_COL
            ]
        ].dropna().set_index(OVERALL_CLASSIFICATION_COL).T.to_dict(orient='records')[0]
    )

    print(classification_chance)

    df_data_exploded = process_and_print(
        label="Sanitize and explode words",
        process=lambda: sanitize_and_explode_words(
            df_data,
            stop_words
        )
    )

    sep()

    df_data_words = process_and_print(
        label="Group data words",
        process=lambda: group_count_words(
            df_data_exploded,
            cols=[DATA_COL]
        )
    )

    probabilities_of_classification = {}
    for classification in df_model_classifications:

        probability = process_and_print(
            label="Calculate probability for " + str(classification),
            process=lambda: calculate_chance_for_classification(
                classification=classification,
                classification_chance=classification_chance[classification],
                df_model=df_model_words,
                df_data=df_data_words
            )
        )
        probabilities_of_classification[classification] = probability

    return probabilities_of_classification
