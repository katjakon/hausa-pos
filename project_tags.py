# It is assumed that a text file with ||| parallel data is given, an alignment for said file and a file with tagged source language.
from collections import Counter, defaultdict
import os
import argparse

from nltk.translate import AlignedSent, Alignment


class POSAlignedSent:

    UNK = "<unk>"

    def __init__(self, words, mots, alignment, tags, lang):
        """
        Args:
            words (list): List of strings that represent a sentence in the target language.
            mots (list): List of strings that represent a sentence in the source language.
            alignment (str): A string in the Pharao format that represents the alignment between words and mots, e.g. "1-0 1-2"
            tags (list): List of strings that represent the tags for the words in mots. Must be the same length as mots.
            lang (str): String that represent the language of the source sentence, e.g. "en" for English.
        """
        self.aligned_sent = AlignedSent(words=words, mots=mots, alignment=Alignment.fromstring(alignment))
        if len(tags) != len(self.source_sentence()):
            raise ValueError("POS sequence has different length than source sentence!")
        self.source_pos = tags
        self.alignment = self.aligned_sent.alignment
        self.lang = lang

    def source_sentence(self):
        return self.aligned_sent.words

    def target_sentence(self):
        return self.aligned_sent.mots

    def language(self):
        return self.lang

    def __repr__(self):
        return str(self.aligned_sent)

    def projected_tags(self):
        """Projects tags from the source sentence onto the target sentence.
        
        Returns: List of tuples that stand for tag and tag confidence.
        """
        projected = [[(self.UNK, 0)] for i in range(len(self.target_sentence()))]
        # Iterate for alignment indices.
        for src_idx, trg_idx in self.alignment:
            # Get associated tag for source language.
            tag = self.source_pos[src_idx]
            # Add tag to aligned target word (One word can be aligned to multiple word in source language.).
            projected[trg_idx].append(tag)
        return projected

class Experiment:

    CORPORA_DIR = "parallel"
    ALIGNMENT_DIR = "aligned"
    AVAIL_ALIGN = ("fast_align", "SimAlign")
    TAGGED_DIR = "tagged"
    TARGET = "ha"
    PARALLEL_EXT = "txt"
    TAGGED_EXT = "tagged"
    ALIGN_EXT = "align"
    PROJECTED_DIR = "experiments"
    UNK = "<unk>"
    ALIGNMENT = {
        "SimAlign": ("inter", "itermax", "mwmf"),
        "fast_align": ("forward", "reverse", "sym")
    }

    def __init__(self, corpora, langs, alignment=None):
        """
        corpora (list): List with corporas that should be used, i.e. ["Tanzil", "Ubunutu"]
        langs (list): List of languaged that should be used
        alignment (dict): Keys are the available alignment methods, value are their respective types, e.g {"SimAlign": "inter"}
        """
        if alignment is None:
            alignment = self.ALIGNMENT
        self.corpora = corpora
        self.langs = langs
        self.alignments = [(key, align_type) for key in alignment for align_type in alignment[key]]
        self.aligned_sentences = defaultdict(list)
        self.lexicon = defaultdict(Counter)
        self.projected_tags = defaultdict(list)
        
        # Corpus statistics
        self.n_src_sents = defaultdict(int)

        self._initialize()

    def _initialize(self):
        "Here, everything is intialized."
        for corp in self.corpora:
            for lang in self.langs:
                file_parallel = "{}-{}-{}.{}".format(corp, lang, self.TARGET,  self.PARALLEL_EXT)
                file_tagged = "{}-{}-{}.{}".format(corp, lang, self.TARGET,  self.TAGGED_EXT)
                corp_path = os.path.join(self.CORPORA_DIR, file_parallel)
                align_paths = [os.path.join(self.ALIGNMENT_DIR, align, "{}-{}-{}-{}.{}".format(corp, lang, self.TARGET,  align_type, self.ALIGN_EXT)) 
                    for align, align_type in self.alignments]
                tagged_path = os.path.join(self.TAGGED_DIR, file_tagged)
                sent_pairs = self._read_parallel_file(corp_path)
                alignments = [self._read_align_file(align_path) for align_path in align_paths]
                tags = self._read_tagged_file(tagged_path)
                for alignment in alignments:
                    for (src, trg), align_strs, tag_seq in zip(sent_pairs, alignment, tags):
                        if not tag_seq:
                            continue
                        try:
                            sent = POSAlignedSent(words=src, mots=trg, alignment=align_strs, tags=tag_seq, lang=lang)
                            self.aligned_sentences[tuple(trg)].append(sent)
                            self.n_src_sents[lang] += 1
                        except ValueError:
                            print("Sentence with different length POS Sequence")
        for sent in self.aligned_sentences:
            self.projected_tags[tuple(sent)] = self._project_tags(sent)

    def tagged_target_sentences(self, infer_lexically=True):
        "Return the target sentences tagged via projection."
        tagged_sents = defaultdict(list)
        for s, tags in self.projected_tags.items():
            if infer_lexically:
                tags = self._infer_lexically(s, tags)
            tagged_sents[s] = tags
        return tagged_sents

    def _infer_lexically(self, target_sentence, tags):
        infered_tags = []
        for idx, t in enumerate(tags):
            new_tag = t
            if t == self.UNK:
                entry = self.lexicon[target_sentence[idx]]
                if entry:
                    for lex_tag, _ in entry.most_common():
                        if new_tag != self.UNK:
                            break
                        new_tag = lex_tag
            infered_tags.append(new_tag)
        return infered_tags

    def get_alignments(self, target_sentence):
        "Return all alignments for a target sentence"
        sent = tuple(target_sentence)
        return self.aligned_sentences[sent]

    def write_tagged_to_file(self, out_dir):
        "Write projected sentences to a file."
        file_path = self.name + ".conll"
        path = os.path.join(out_dir, file_path)
        projected_dict = self.tagged_target_sentences()
        with open(path, "w", encoding="utf-8") as file:
            for sent, tags in projected_dict.items():
                if any(tag == self.UNK for tag in tags):
                    continue
                for word, tag in zip(sent, tags):
                    file.write("{}\t{}\n".format(word, tag))
                file.write("\n")

    @property
    def name(self):
        if len(self.alignments) > 1:
            align_str = "multi-align"
        else:
            align, align_type = self.alignments[0]
            align_str = "{}_{}".format(align, align_type)
        return "{}-{}-{}".format(
            "_".join(self.corpora), 
            "_".join(self.langs),
            align_str)

    def _project_tags(self, target_sentence):
        "Project tag onto target language"
        sent = tuple(target_sentence)
        # Get all sentences that have been aligned to target sentence.
        aligned_sents = self.aligned_sentences[sent]
        # Stores the probability of tags for each target word.
        projected_tags_counters = [defaultdict(int) for i in range(len(sent))]
        for pos_sent in aligned_sents:
            # Get projected tags for each aligned sentence.
            projected_tags = pos_sent.projected_tags()
            # Iterate over possible tags and sum up probability of tag for each word.
            for idx, poss_tags in enumerate(projected_tags):
                for tag, score in poss_tags:
                    projected_tags_counters[idx][tag] += score
        final_tags = []
        for counter in projected_tags_counters:
            if counter:
                most_prob = max(counter, key= lambda x: counter[x])
            else:
                most_prob = self.UNK
            final_tags.append(most_prob)
        # Count 
        for idx, t in enumerate(final_tags):
            self.lexicon[sent[idx]][t] += 1
        return final_tags

    def _read_parallel_file(self, file_name):
        pairs = []
        with open(file_name, encoding="utf-8") as file:
            for line in file:
                line = line.strip().split("|||")
                src, trg = line
                pairs.append((src.split(), trg.split()))
        return pairs

    def _read_align_file(self, file_name):
        "Reads in a alignment file in the Pharao format."
        alignments = []
        with open(file_name, encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                alignments.append(line)
        return alignments

    def _read_tagged_file(self, file_name):
        "Reads in a file that contains the tags for source sentences."
        tags = []
        with open(file_name, encoding="utf-8") as file:
            for line in file:
                tag_seq = []
                line = line.strip().split()
                tag_seq = [(tag_score.split("-")[0], float(tag_score.split("-")[1])) for tag_score in line]
                tags.append(tag_seq)
        return tags

    def vocabulary(self):
        "Returns a set with all words that appear in the corpus."
        vocab = set()
        for sent, tags in self.tagged_target_sentences().items():
            if any(tag == self.UNK for tag in tags):
                continue
            vocab = vocab.union(sent)
        return vocab

if __name__ == "__main__":
    CORPUS = ["Tanzil"]
    parser = argparse.ArgumentParser(description="Annotation Projection")
    parser.add_argument("langs", help="Abbreviation for source languages seperated by commas, e.g. ar,en")
    parser.add_argument("align", help="Alignment methods, either SimAlign or fast_align.")
    parser.add_argument("type", help="Alignment type. For SimAlign: inter, itermax or mwmf. For fast_align: forward, reverse or sym")
    parser.add_argument("out", help="Output directory.")
    args = parser.parse_args()
    langs = sorted(args.langs.split(","))
    alignment = {args.align: [args.type]}
    print("Projecting tags...")
    experiment = Experiment(CORPUS, langs, alignment)
    print("Writing file {}.conll to directory {}".format(experiment.name, args.out))
    experiment.write_tagged_to_file(args.out)

