import numpy as np
import scipy.linalg as scipy_linalg

class one_sentence_embedding():

    def __init__(self, weighting, w2v_model, dimension):
        self.bm25_weighting = weighting
        self.model = w2v_model
        self.dimension = dimension

    def sentence_to_vector(self, sentence):
        res = np.zeros((self.dimension))
        counter = 0
        curr_weighting = self.bm25_weighting.get_bm25_weight(sentence)
        for word in curr_weighting:
            if word in self.model:
                counter += 1
                res += self.model[word] * curr_weighting[word]
        return res.tolist()

    def get_similrity(self, sentence1, sentence2):
        v1 = self.get_sentence_to_vector(sentence1)
        v2 = self.get_sentence_to_vector(sentence2)
        res = 0
        if v1[0] > 0 and v2[0] > 0:
            word1_vector = v1[1]
            word2_vector = v2[1]
            res = np.dot(word1_vector, word2_vector) / (scipy_linalg.norm(word1_vector) * scipy_linalg.norm(word2_vector))
        print("The similarity of " + sentence1 + " and " + sentence2 + " is " + str(res))