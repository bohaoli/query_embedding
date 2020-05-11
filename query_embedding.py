import time
import gensim

from flask import Flask, request
from one_sentence_embedding import one_sentence_embedding
from bm25_weighting import bm25_weighting
from preprocessing_utils import preprocessing_utils
from write_float import document_embedding_date
from query_trend import query_trend

# http://127.0.0.1:5000
app = Flask(__name__)
# These are some metadata of our word embedding
model_file = "cord19-300d.bin" # The file that we'll load the word embedding
dimension = 300 # The dimension of the word embedding
idf_para_shelve = "idf_score_para" # The name of the shelve of idf score for each word against paragraphs. For paragraph embedding
idf_doc_shelve = "idf_score_doc" # The name of the shelve of idf score of each word against articles. For query trend.
document_embedding_and_date = "binary_file.bin" # A binary file of embeddings of all documents of our raw data and their publish time.

# load the word2vec model
start = time.time()
print("Start to load bm25 weightings of our corpus based on paragraph")
bm25wei_para = bm25_weighting(idf_para_shelve, 1.2, 0.75, preprocessing_utils(), None)
print("Start to load bm25 weightings of our corpus based on article")
bm25wei_doc = bm25_weighting(idf_doc_shelve, 1.2, 0.75, preprocessing_utils(), None)
print("Start to load a " + str(dimension) + " dimension word embedding model from file " + model_file)
w2v_model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)
embedding_date_list = document_embedding_date("document_embedding_300d-data", document_embedding_and_date).vector_date_list
ose_para = one_sentence_embedding(bm25wei_para, w2v_model, dimension)
ose_doc = one_sentence_embedding(bm25wei_doc, w2v_model, dimension)
qt = query_trend(embedding_date_list, ose_doc)
end = time.time()
print("It takes " + str(end - start) + " seconds to load the bm25 weightings and the model")


@app.route("/")
def query():
    return 'query_page.html'

# This takes a pair of param whose key is "query" and value is "<any query you want>"
# It will return a dictionary whose key is "vector" and value is a list of floats that stands for
# the embedding of the query
@app.route("/results") #, methods=['POST']
def results():
    global ose_para
    """Generate a result set for a query and present the 10 results starting with <page_num>."""
    query = request.args.get("query")
    res = {}
    res["vector"] = ose_para.sentence_to_vector(query)
    return res

# This takes a pair of param whose key is "query" and value is "<any query you want>"
# It will return the file name of a picture that shows the trend of research papers about a given query
# Our front end will show you this picture
@app.route("/trend") #, methods=['POST']
def trend():
    global qt
    """Generate a result set for a query and present the 10 results starting with <page_num>."""
    query = request.args.get("query")
    res = {}
    res["url"] = qt.plot_trend(query, 1000)
    return res

# This is the main funtion
if __name__ == "__main__":
    app.run()