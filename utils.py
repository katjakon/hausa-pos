# Utils


def read_train_sentences(file_path):
    "Read a tsv file where first column is token and second is tag."
    sentences = []
    curr_sent = []
    with open(file_path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            line = line.split("\t")
            if len(line) == 2:
                token, ud_tag = line
                curr_sent.append((token, ud_tag))
            else:
                sentences.append(curr_sent)
                curr_sent = []
    return sentences



def read_test_sentences(file_path):
    "Read a tsv file where first column is token, second is hausa tag and third is ud tag."
    sentences = []
    curr_sent = []
    with open(file_path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            line = line.split("\t")
            if len(line) == 3:
                token, hausa_tag, ud_tag = line
                curr_sent.append((token, hausa_tag, ud_tag))
            else:
                sentences.append(curr_sent)
                curr_sent = []
    return sentences


def write_out(sentences, path):
    with open(path, "w", encoding="utf-8") as file:
        for sent in sentences:
            for row in sent:
                str_row = "\t".join(row)
                file.write(str_row+"\n")
            file.write("\n")
