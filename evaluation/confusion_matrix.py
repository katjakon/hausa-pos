# Confusion Matrix
import os

import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
    y_test = res_dict[exp][GOLD_COL]
    y_pred = res_dict[exp][3]
    conf_m = pd.crosstab(y_test, y_pred, rownames=['True Tags'], colnames=['Predicted Tags'], normalize="index")
    conf_m_abs = pd.crosstab(y_test, y_pred, rownames=['True Tags'], colnames=['Predicted Tags'])
    if "SCONJ" not in conf_m:
        conf_m["SCONJ"] = np.zeros(conf_m.shape[0])
        conf_m_abs["SCONJ"] = np.zeros(conf_m.shape[0])
    if "X" not in conf_m:
        conf_m["X"] = np.zeros(conf_m.shape[0])
        conf_m_abs["X"] = np.zeros(conf_m.shape[0])
    # Reorder
    conf_m.sort_index(axis=1, inplace=True)
    conf_m_abs.sort_index(axis=1, inplace=True)
    annot_labels = conf_m_abs.copy()
    annot_mask = annot_labels == 0
    annot_labels[annot_mask] = '' 
    fig, ax = plt.subplots()
    sns.heatmap(conf_m, annot=annot_labels, fmt="", ax=ax, cmap=sns.light_palette("#3268a8", as_cmap=True), robust=True, cbar_kws={"label": "Counts normalised over True Tags "})
    ax.set(title=exp)
    plt.tight_layout()
    plt.savefig("test.png", format="png")
    plt.show()