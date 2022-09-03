# HMM tagger
from tag_set_mappings import hausa_mapping
from utils import read_test_sentences, read_train_sentences, write_out

DIR = "projected"
TEST = "test_data/revised_test_data.tsv"
TRAIN = os.listdir(DIR)
OUT  = "test_data/predictions"

test_path = os.path.join(TEST)
test_data = read_test_sentences(test_path)

for train_file in TRAIN:
    train_path = os.path.join(DIR, train_file)
    train_data = read_train_sentences(train_path)
    hmm = HiddenMarkovModelTagger.train(labeled_sequence=train_data)

    predict_sentences = []
    for sent in test_data:
        tokens = [token for token, _, _ in sent]
        predicted_tags = predict(tokens, hmm)
        pred_sent = [(token, hau_tag, ud_tag, pred) for (token, hau_tag, ud_tag), pred in zip(sent, predicted_tags)]
        predict_sentences.append(pred_sent)
    result_file = "{}_hmm.tsv".format(train_file.split(".")[0])
    out_path = os.path.join(OUT, result_file)
    write_out(predict_sentences, out_path)
    print(result_file)
