
from pandas import DataFrame

from util import *


def calculate_chance() -> DataFrame:
    raise NotImplementedError()


def test_model(
    model: DataFrame,
    data: DataFrame,
    stop_words: list[str]
) -> DataFrame:

    # TODO: Implement
    # Get overall classification chance
    classification_chance = model[[CLASSIFICATION_COL_FMT], [CHANCE_COL_FMT]]

    print(classification_chance)

    raise NotImplementedError()
    # Use overall word classifications
