# Baseline 
import os
from collections import Counter

from nltk.tag import UnigramTagger

from tag_set_mappings import hausa_mapping
from utils import read_test_sentences, read_train_sentences, write_out

DIR = "projected"
TEST = "test_data/revised_test_data.tsv"
TRAIN = os.listdir(DIR)
OUT  = "test_data/predictions"


def predict(sentence, model, unknown="<unk>"):
    return [tag if tag is not None else unknown for _, tag in model.tag(tokens)]

def get_most_frequent_tag(sentences):
    count_dict = Counter()
    for sent in sentences:
        for token, tag in sent:
            count_dict.update(tag)
    return count_dict.most_common(1)[0][0]


test_path = os.path.join(TEST)
test_data = read_test_sentences(test_path)

for train in TRAIN:
    print(train)
    train_path = os.path.join(DIR, train)
    data = read_train_sentences(train_path)
    most_frequent = hausa_mapping.mapping[get_most_frequent_tag(data)]
    tagger = UnigramTagger(data)
    predict_sentences = []
    for sent in test_data:
        tokens = [token for token, _, _ in sent]
        predicted_tags = predict(tokens,tagger, most_frequent)
        pred_sent = [(token, hau_tag, ud_tag, pred) for (token, hau_tag, ud_tag), pred in zip(sent, predicted_tags)]
        predict_sentences.append(pred_sent)
    result_file = "{}_baseline.tsv".format(train.split(".")[0])
    out_path = os.path.join(OUT, result_file)
    write_out(predict_sentences, out_path)
    print(result_file)