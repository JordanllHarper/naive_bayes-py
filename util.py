
def format_overall_col(col):
    return "{col}_{suffix}".format(col=col, suffix=SUFFIX_OVERALL)


CLASSIFICATION_COL = "classification"
DATA_COL = "text"
SUFFIX_OVERALL = "overall"

OVERALL_CLASSIFICATION_COL = format_overall_col(CLASSIFICATION_COL)
OVERALL_COUNT_COL = format_overall_col("count")
OVERALL_CHANCE_COL = format_overall_col("chance")
OVERALL_PERCENT_COL = format_overall_col("%")


def read_stop_words(stop_words_path: str) -> list[str]:
    stop_words = open(stop_words_path, "r").readlines()
    stop_words = list(map(lambda x: x.strip().lower(), stop_words))
    return stop_words
