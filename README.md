# Hausa Part-of-Speech Tagging in a Low-Resource Scenario
This project is for my Bachelor thesis at the University of Potsdam for which I investigated automatic POS Tagging in Hausa by assuming that no annotated data is available. Instead, I utilized parallel sentences in Englisch, French, Arabic and German to induce word classes.

## Overview
Parallel sentences for the Quran can be found in the directory `parallel`. Each file has alignment information in the directory `aligned` and tags for the source language in the directory `tagged`.<br>
Alignments and source tags are used to project tags onto the Hausa sentences. These files can be found in the directory `projected`. On the data in `projected` I train different tagging models. 
I evaluate them on the test data in the directory `test_data`. Predictions for each tagger can also be found in this directory under `predictions`.
All the figure can be found in the directory `figures`.

## Notebooks
For various aspects of this work, I made use of Jupyter notebooks in Google Colab. They can be found in the directory `notebooks`:
+ In `word_alignment_hausa.ipynb`, parallel sentences are aligned with `fast_align` and `SimAlign`
+ In `tag_english.ipynb`, `tag_multilingual.ipynb` and `tag_arabic.ipynb`, POS tagging for the source languages is done
+ In `bi-lstm-trainer.ipynb`, a BI-LSTM model with a CRF layer is trained.

See the individual notebooks for instructions on how to use them.

## Annotation Projection
After word alignment and tagging of the source languages, the tags of the source language are projected onto the Hausa sentences. 
This is can be reproduced with `project_tags.py`. It takes the following arguments:
+ `langs`: These are the source languages that should be used. The following languages are available `ar` (Arabic), `en` (English), `de` (German) and `fr` (French). If multiple source languages should be used, seperate them with a comma. 
+ `align`: This is the alignment method that should be used. Can either be `SimAlign` or `fast_align`
+ `type`: This is the type of alignment. For `SimAlign`, this can either be `inter`, `itermax` or `mwmf`. For `fast_align`, this can be `forward`, `reverse` or `sym`
+ `out`: Directory where the results should be stored.

For example:
`project_tags.py  ar,en  SimAlign itermax results/`

## Train Tagger
In the files `baseline.py` and `hmm`, the Unigram and Hidden Markov Models are trained and predictions for the test set are written to files. 

## Evaluation
+ for accuracy values, execute `python evaluation/accuracy.py`
+ for confusion matrices, execute `python evaluation/confusion_matrix.py`


