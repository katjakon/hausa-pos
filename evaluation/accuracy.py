# Accuracy
import os

import pandas as pd

RESULTS = os.path.join("test_data/predictions/")
result_files = os.listdir(RESULTS)

GOLD_COL = 2

res_dict = dict()
accuracy = dict()


# Read in all files with prediction and store data frame in dictionary.
for file in result_files:
    path = os.path.join(RESULTS, file)
    df = pd.read_csv(path, sep="\t", header=None, na_values="NULL", keep_default_na=False)
    res_dict[file] = df

for exp in res_dict:
    results = res_dict[exp]
    n_cols = results.shape[1]
    correct = (results[GOLD_COL] == results[3]).sum()
    if n_cols >= 5:
        correct2 = (results[GOLD_COL] == results[4]).sum()
        correct_avg = (correct/results.shape[0] + correct2/results.shape[0])/2
        accuracy[exp] = correct_avg
    else:
        correct_avg = correct/results.shape[0]
        accuracy[exp] = correct_avg

for file in accuracy:
    print(file, accuracy[file])