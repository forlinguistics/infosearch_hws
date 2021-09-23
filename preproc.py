import numpy as np
from main import preproc
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
corpus = []
curr_dir = os.getcwd()
for root, dirs, files in os.walk(curr_dir + '/friends-data/'):
    for name in files:
        with open(os.path.join(root, name), 'r', encoding='UTF-8') as f:
            corpus.append(preproc(f.read()))
vectorizer = TfidfVectorizer()
matr = vectorizer.fit_transform(corpus).toarray()
np.save('matr.npy', matr)
with open('tfidf.pk', 'wb') as fin:
    pickle.dump(vectorizer, fin)