import numpy as np
import time
import os
import json
import datetime
import scipy.linalg as scipy_linalg
import heapq
import matplotlib.pyplot as plt
import gensim

from bm25_weighting import bm25_weighting
from write_float import document_embedding_date
from preprocessing_utils import preprocessing_utils

class query_trend():

    def __init__(self, embedding_date_list, one_sentence_embedding):
        self.embedding_date_list = embedding_date_list
        self.ose = one_sentence_embedding


    def get_similrity(self, v1, v2):
        similarity = np.dot(v1, v2) / (scipy_linalg.norm(v1) * scipy_linalg.norm(v2))
        return similarity

    def get_trend(self, query, top_N):
        trend = []
        query_vector = np.asarray(self.ose.sentence_to_vector(query))
        for pair in self.embedding_date_list:
            heapq.heappush(trend, (-self.get_similrity(query_vector, pair[0]), pair[1]))

        res = []
        counter = 0
        while counter < top_N:
            pair = heapq.heappop(trend)
            res.append((-pair[0], pair[1]))
            counter += 1
        return res

    def plot_trend(self, query, top_N):
        points = self.get_trend(query, top_N)
        x_val = [1970 + pair[1] / 86400 / 365 for pair in points]
        y_val = [pair[0] for pair in points]

        fig = plt.figure()
        plt.scatter(x_val, y_val)
        fig.suptitle(query, fontsize=15)
        plt.xlabel('Time', fontsize=10)
        plt.ylabel('Similarity', fontsize=10)
        filename = query.rstrip().replace(" ", "_") + '_.png'
        fig.savefig(filename)
        return os.getcwd() + "\\" + filename