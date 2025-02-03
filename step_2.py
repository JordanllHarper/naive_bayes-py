import pandas as pd
from naive_bayes import space, process_and_print


def get_num_words_per_classification(df, classification_column):
    return df.groupby(
        classification_column,
        observed=True
    ).size().reset_index(name="count").set_index(classification_column).T.to_dict(orient='records')[0]


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
    space()

    # Dataframe structure:
    # Word | Classification 1 Count | Classification 2 Count | ...

    words_filtered = process_and_print(
        label="Words filtered",
        process=lambda:
            words[~words[data_column].isin(stop_words)]
    )

    words_grouped = process_and_print(
        label="Word groups",
        process=lambda: words_filtered.groupby(
            # type: ignore
            [
                classification_column,
                data_column
            ],
            observed=True,
            sort=False,
        ).size().reset_index(name="count")
    )

    print("Naturally categorized")

    print(words_grouped[words_grouped["text"].str.contains("naturally")])

    num_words_per_classification = process_and_print(
        label="Number of words per classification",
        process=lambda: get_num_words_per_classification(
            words_grouped,
            classification_column=classification_column,
        )
    )

    words_grouped["chance"] = process_and_print(
        label="Chance calculated",
        process=lambda: words_grouped["count"] /
        words_grouped[classification_column].map(
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
