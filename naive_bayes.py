import pandas as pd

import sys


def sep():
    print()


def step_print(step_num: int, message: str):
    print("--------")
    print("[STEP {step_num}]: {msg}\n".format(step_num=step_num, msg=message))


step_print(0, "Setup")

args = sys.argv[1:]

path_to_csv = args[0]
print("CSV path:", path_to_csv)
stop_words_path = args[1]
print("Stop words path:", stop_words_path)
test_data = args[2]
print("Test data path:", test_data)

df = pd.read_csv(path_to_csv)
stop_words: list = open(stop_words_path, "r").readlines()
stop_words.extend

sep()

data = df.columns[0]
print("Data header:", data)
classification = df.columns[1]
df[classification] = df[classification].astype("category")
print("Classification header:", classification)
classification_categories = df[classification].unique()
print("Classification categories")
print(classification_categories)

print(df)

num_records = df[data].count()

sep()

print("Loaded:", num_records, "records")

step_print(1, "Overall classifications")

df_overall_classification_chance = pd.DataFrame(
    {
        "Classification": df[classification].unique(),
        "Count": df[classification].value_counts()
    }
)
df_overall_classification_chance["Chance"] = df_overall_classification_chance["Count"] / num_records
df_overall_classification_chance["%"] = df_overall_classification_chance["Chance"] * 100

print(df_overall_classification_chance)

# ---

step_print(2, "Map words and their classification counts")

# Sanitization of words


def sanitize_words(df: pd.DataFrame):
    df[data] = df[data].str.split().transform(
        lambda l: list(map(lambda w: w.strip().lower(), l))
    ).transform(
        lambda l: list(filter(lambda w: [w for c in w if c.isalpha()], l))
    )

    return df


words = sanitize_words(df)
print(words)

# Dataframe structure:
# Word | Classification 1 Count | Classification 2 Count | ...

words_flattened = words.explode(str(data))
print(words_flattened)

words_filtered = words_flattened[~words_flattened[data].isin(stop_words)]
print(words_filtered)

# Code adapted from Hait (2021)
# words_groups = words_filtered[words_filtered.duplicated(keep=False)].copy() print(words_groups)
#
# words_count = words_groups.value_counts().reset_index(name='count')
# print(words_count)

# words_counted = words_flattened.groupby(df.columns)
# print(words_counted)
# word_classification_count = pd.DataFrame(
#     columns=["word"] + list(map(lambda x: str(x), classification_categories)),
#     # data = each word
# )
# print(word_classification_count)
