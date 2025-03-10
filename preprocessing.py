from pandas import DataFrame

from train import standardize_column_names

from util import *


def preprocess(df: DataFrame, data_col_index, class_col_index):
    df = standardize_column_names(df, data_col_index, class_col_index)

    classification_categories = df[CLASSIFICATION_COL].unique()
    data_df = []
    test_df = []
    for category in classification_categories:
        category_data = df[df[CLASSIFICATION_COL] == category]

        data = category_data.sample(frac=.8)
        data_df.append(data)

        test = category_data.drop(data.index)
        test_df.append(test)

    return pd.concat(data_df), pd.concat(test_df)
