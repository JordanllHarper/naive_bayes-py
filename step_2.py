import pandas as pd
from naive_bayes import sep


def get_num_words_per_classification(df, classification_column):
    print("Num words per classification")
    sums = df.groupby(
        classification_column,
        observed=True
    ).size().reset_index(name="count").set_index(classification_column).T.to_dict(orient='records')[0]

    print(sums)
    return sums


def map_words_to_classification_counts(
        start_df: pd.DataFrame,
        data_column: str,
        stop_words: list,
        classification_column: str
):

    def sanitize_and_explode_words(df: pd.DataFrame):
        df[data_column] = df[data_column].str.split().transform(
            lambda l: list(map(lambda w: w.strip().lower(), l))
        ).transform(
            lambda l: list(filter(lambda w: [w for c in w if c.isalpha()], l))
        )
        df = df.explode(str(data_column))

        return df

    words = sanitize_and_explode_words(start_df)
    print(words)
    sep()

    # Dataframe structure:
    # Word | Classification 1 Count | Classification 2 Count | ...

    words_filtered = words[~words[data_column].isin(stop_words)]

    print("Words filtered")
    print(words_filtered)

    sep()

    print("Word groups")

    words_grouped = words_filtered.groupby(
        # type: ignore
        [
            classification_column,
            data_column
        ],
        observed=True,
        sort=False,
    ).size().reset_index(name="count")

    print(words_grouped)

    sep()

    num_words_per_classification = get_num_words_per_classification(
        words_grouped,
        classification_column=classification_column,
    )
    sep()

    print("Naturally categorized")

    print(words_grouped[words_grouped["text"].str.contains("naturally")])

    sep()

    print(num_words_per_classification)

    sep()

    words_grouped["chance"] = words_grouped["count"] / \
        words_grouped[classification_column].map(
            num_words_per_classification
    ).astype(int)

    print(words_grouped)

    words_grouped["%"] = words_grouped["chance"] * 100
    words_grouped = words_grouped.sort_values(
        "%",
        ascending=False,
    )

    print(words_grouped)

    sep()

    print(words_grouped[words_grouped["spam"].astype(int) == 1])
    sep()

    return words_grouped
