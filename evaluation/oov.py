# OOV vs In Vocabulary
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from project_tags import Experiment


RESULTS = os.path.join("test_data/predictions/")
result_files = os.listdir(RESULTS)

# Specify for which source languages.
LANGS = ["ar", "en", "de", "fr"]
LANGS.sort()

# Specify which alignment method and which alignment type
ALIGNMENT = "SimAlign"
TYPE = "itermax"

prefix = "Tanzil-{}-{}_{}".format("_".join(LANGS), ALIGNMENT, TYPE)

GOLD_COL = 2

res_dict = dict()
accuracy = dict()


# Read in all files with prediction and store data frame in dictionary.
for file in result_files:
    path = os.path.join(RESULTS, file)
    df = pd.read_csv(path, sep="\t", header=None, na_values="NULL", keep_default_na=False)
    res_dict[file] = df

project = Experiment(["Tanzil"], LANGS, {ALIGNMENT: [TYPE]})
vocab = project.vocabulary()
data = pd.DataFrame(data={"Accuracy": [], "kind": [], "Model": []})

for ex in res_dict:
    if ex.startswith(prefix):
        print(ex)
        results = res_dict[ex]
        oov_mask = results[0].apply(lambda x: x not in vocab)
        invocab_mask = ~oov_mask
        oov_words = results[oov_mask]
        invocab_words = results[invocab_mask]
        acc_oov = (oov_words[GOLD_COL] == oov_words[3]).sum() / oov_words.shape[0]
        acc_incovab = (invocab_words[GOLD_COL] == invocab_words[3]).sum() / invocab_words.shape[0]
        model = "BI-LSTM"
        if "baseline" in ex:
            model = "Unigram"
        elif "hmm" in ex:
            model = "HMM"
        new_rows = pd.DataFrame({"Accuracy": [acc_oov, acc_incovab], "Model": [model, model], "kind": ["Out of Vocabulary", "In Vocabulary"]})
        data = pd.concat([data, new_rows])

ax = sns.catplot(x="kind", y="Accuracy", hue="Model",
            data=data,
            kind="bar", legend=True, orient="v", palette=sns.color_palette("rocket_r"), height=4, aspect=0.7, legend_out=True)
ax.set(xlabel='', ylabel='Accuracy')
plt.show()
