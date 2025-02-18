
from pandas import DataFrame

from train import sep
from util import *


def calculate_chance(df: DataFrame) -> DataFrame:
    raise NotImplementedError()


def test_model(
    model: DataFrame,
    data: DataFrame,
    stop_words: list[str]
) -> DataFrame:

    sep()

    print(model)
    # TODO: Implement
    # Get overall classification chance
    classification_chance = model[[
        OVERALL_CLASSIFICATION_COL, OVERALL_CHANCE_COL]].dropna()

    print(classification_chance)

    raise NotImplementedError()
