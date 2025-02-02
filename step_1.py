import pandas as pd
from pandas import DataFrame


def step_one(df: DataFrame, classification, num_records):
    df_overall_classification_chance = pd.DataFrame(
        {
            "Classification": df[classification].unique(),
            "Count": df[classification].value_counts()
        }
    )
    df_overall_classification_chance["Chance"] = df_overall_classification_chance["Count"] / num_records
    df_overall_classification_chance["%"] = df_overall_classification_chance["Chance"] * 100

    print(df_overall_classification_chance)
