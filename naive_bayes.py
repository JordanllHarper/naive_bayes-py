import pandas as pd

import sys


def step_print(step_num: int, message: str):
    print("--------")
    print("[STEP {step_num}]: {msg}\n".format(step_num = step_num, msg=message))



step_print(0, "Setup")

args = sys.argv[1:]

path_to_csv = args[0]
print("CSV path:", path_to_csv)
stop_words = args[1]
print("Stop words path:", stop_words)
test_data = args[2]
print("Test data path:", test_data)

df = pd.read_csv(path_to_csv)

print()

data = df.columns[0]
print("Data header:", data)
classification = df.columns[1]
print("Classification header:", classification)

print(df)

num_records = df[data].count()

print()
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

step_print(2, "Map words and their classification counts")

# Dataframe structure:
# Word
# Classification
# Count of word

words = df[data].str.split().transform(lambda series: list(filter(lambda word: word != " ", series)))

# Series of words
# Establish a count of each word

print(words)