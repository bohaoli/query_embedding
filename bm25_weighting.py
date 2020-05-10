import shelve

from preprocessing_utils import preprocessing_utils
from collections import defaultdict

class bm25_weighting():

    def __init__(self, idf_file, k_value, b_value, utils, idf_object):
        self.preprocessing_utils = utils
        self.idf_score = {}
        if idf_object != None:
            self.idf_score = idf_object.idf
        else:
            self.idf_score = shelve.open(idf_file)# It takes time. We'd better run the preprocessing method of an idf_score object before hand
        self.average_doc_length = self.idf_score[utils.ave_doc_len_name]
        self.k = k_value
        self.b = b_value

    def best_match(self, cwd, len_d):
        return (self.k + 1) * cwd / (cwd + self.k * (1 - self.b + self.b * len_d / self.average_doc_length))

    def get_bm25_weight(self, document):
        word_list = self.preprocessing_utils.my_tokenizer(document)
        word_freq = defaultdict(lambda: 0)
        total_num_words = 0
        for word in word_list:
            if word in self.idf_score:
                total_num_words += 1
                word_freq[word] += 1
        for word in word_freq:
            word_freq[word] /= total_num_words
        for word in word_freq:
            word_freq[word] = self.idf_score[word] * self.best_match(word_freq[word], total_num_words)
        return word_freq