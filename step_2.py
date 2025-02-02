import pandas as pd


def step_two(start_df: pd.DataFrame, data_column: str, stop_words: list, classification_column: str):

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

    # Dataframe structure:
    # Word | Classification 1 Count | Classification 2 Count | ...

    print("Words filtered")
    words_filtered = words[~words[data_column].isin(stop_words)]
    print(words_filtered)
    print()

    print("Word groups")
    words_groups = words_filtered.groupby(
        [
            classification_column,
            data_column
        ],
        observed=True,
        sort=False,
        as_index=True,
    ).size().to_frame("count")
    # the count of the classification which we don't know yet
    words_groups["chance"] = words_groups["count"] / 100
    print(words_groups)
    print()

    # groups_standalone = words_groups.groups
    #
    # print()
    #
    # print("Groups standalone")
    # print(groups_standalone)

    # Sanitization of words
