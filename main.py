import os
import string
from natasha import Doc, MorphVocab, Segmenter
import numpy as np
import pymorphy2
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import spatial
import pickle


def preproc(text):
    morph = pymorphy2.MorphAnalyzer()
    morph_vocab = MorphVocab()
    segmenter = Segmenter()
    text_without_punct = text.translate(str.maketrans('', '', string.punctuation))
    doc = Doc(text_without_punct[1:])
    doc.segment(segmenter)
    tokens = []
    for token in doc.tokens:
        tokens.append(morph.parse(token.text)[0].normal_form)
    lemmatized_text = ' '.join(tokens).lower()
    return (lemmatized_text)


def query_proc(query, corp_matr):
    vectorizer = pickle.load(open("tfidf.pk", 'rb'))
    query_vec = vectorizer.transform([preproc(query)]).toarray()
    cosine_sims = np.apply_along_axis(lambda x: sim(x, query_vec), 1, corp_matr)
    return (cosine_sims)


def sim(vec1, vec2):
    return (1 - spatial.distance.cosine(vec1, vec2))


def tf_idf_sims(query):
    f_names = []
    curr_dir = os.getcwd()
    for root, dirs, files in os.walk(curr_dir + '/friends-data/'):
        for name in files:
            f_names.append(name)
    matr = np.load('matr.npy')
    a = list(zip(f_names, query_proc(query, matr)))
    a.sort(key=lambda x: x[1], reverse=True)
    return (a)


if __name__ == "__main__":
    vectorizer = TfidfVectorizer()
    inp_string = input()
    e_list = tf_idf_sims(inp_string)
    for elem in e_list:
        print(elem[0] + '\t' + str(elem[1]))
