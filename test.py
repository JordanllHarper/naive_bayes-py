
from pandas import DataFrame

from formatting import process_and_print
from train import sep
from util import *


def test_model(
    df_model: DataFrame,
    df_data: DataFrame,
    stop_words: list[str]
) -> DataFrame:

    df_data = standardize_column_names(df_data)
    sep()

    print(df_model)
    classification_chance = df_model[[
        OVERALL_CLASSIFICATION_COL, OVERALL_CHANCE_COL]].dropna()

    print(classification_chance)

    words = process_and_print(
        label="Sanitize and explode words",
        process=lambda: sanitize_and_explode_words(
            df_data, stop_words
        )
    )

    sep()

    words_grouped = process_and_print(
        label="Word groups",
        process=lambda: group_words(words)
    )

    raise NotImplementedError()
