# Ambigous Words
import sys
import os
from collections import defaultdict

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
        model = "BI-LSTM"
        if "baseline" in ex:
            model = "Unigram"
        elif "hmm" in ex:
            model = "HMM"
        word_count = results[0].value_counts()
        ambig_dict = defaultdict(set)
        for word in word_count.index:
            word_mask = results[0].apply(lambda x: x == word)
            word_tags = results[word_mask]
            ambig_dict[word].update(word_tags[2])
        ambig_words = [word for word in ambig_dict if len(ambig_dict[word]) > 1]
        # print(round(100*(len(ambig_words)/word_count.shape[0]),2))
        ambig_mask = results[0].apply(lambda x: x in ambig_words)
        unamig_mask = ~ambig_mask
        ambig = results[ambig_mask]
        unambig = results[unamig_mask]
        acc_ambig = (ambig[GOLD_COL] == ambig[3]).sum() / ambig.shape[0]
        acc_unambig = (unambig[GOLD_COL] == unambig[3]).sum() / unambig.shape[0]
        new_rows = pd.DataFrame({"Accuracy": [acc_ambig, acc_unambig], "Model": [model, model], "kind": ["Ambiguous", "Unambiguous"]})
        data = pd.concat([data, new_rows])

ax = sns.catplot(x="kind", y="Accuracy", hue="Model",
            data=data,
            kind="bar", legend=True, orient="v", palette=sns.color_palette("rocket_r"), height=4, aspect=0.7, legend_out=True)
ax.set(xlabel='', ylabel='Accuracy')
plt.show()
