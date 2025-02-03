import pandas as pd
from pandas import DataFrame


def get_classification_counts(df, classification_column):
    return pd.DataFrame(
        {
            "classification": df[classification_column].unique(),
            "count": df[classification_column].value_counts()
        }
    )


def get_overall_classification_chance(df_overall_classification_counts: DataFrame, num_records):
    df_overall_classification_chance = df_overall_classification_counts.copy()

    df_overall_classification_chance["chance"] = df_overall_classification_chance["count"] / num_records
    df_overall_classification_chance["%"] = df_overall_classification_chance["chance"] * 100

    return df_overall_classification_chance
