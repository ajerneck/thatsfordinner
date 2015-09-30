"""Different ways of evaluating the topic models."""

import collections
from gensim import models
from gensim.matutils import Scipy2Corpus
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import psycopg2
import scipy.sparse
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import explore as e
import production as p

from evaluate_functions import *

df = p.load_data()
df = p.clean_formatting(df)
df = p.remove_stopwords(df)

assert(type(df)) is pd.core.frame.DataFrame, "%r is not a DataFrame." % df
assert(df.shape) == (16526, 7), "Has the wrong shape."

vectorizer = CountVectorizer(
    stop_words=p.get_stop_words()
    , ngram_range=(1, 1)
    , token_pattern = '[A-Za-z]+'
)
search_cols = df['ingredient_txt_no_stopwords'].str.cat(df['title'].values, sep=' ')
features = vectorizer.fit_transform(search_cols)

## transform document - term matrix to corpus.

## reconstruct documents using tokens from the vectorizer.
iz,jz,vz = scipy.sparse.find(features)
words = collections.defaultdict(list)
for i,j in zip(iz, jz):
    words[i].append(j)
vocab = np.asarray(vectorizer.get_feature_names())
for k,v in words.iteritems():
    words[k] = vocab[v].tolist()

words_docs = collections.defaultdict(list)
for k,v in words.iteritems():
    for w in v:
        words_docs[w].append(k)


## transform reconstructed documents into a corpus.
texts = [v for v in words.itervalues()]
corpus = [models.doc2vec.TaggedDocument(t, [i]) for (i, t) in enumerate(texts)]

## train model.
m = models.Doc2Vec(corpus)

## search:
p = ['brisket','honey']

x = 'brisket, honey, -chicken'
def parse_pos_neg(string):
    pos, neg = [], []
    for w in [w.strip('.,_* ') for w in string.split()]:
        if w.startswith('-'):
            neg.append(w.strip('-'))
        else:
            pos.append(w)
    return pos, neg


def most_similar_docs(model, words_docs, pos, neg, topwords=20, topdocs=10):

    sims = model.most_similar(pos, neg)

    matches = model.most_similar_cosmul(pos, neg, topn=topwords)

    docs = collections.defaultdict(float)
    for w,s in matches:
        for d in words_docs[w]:
            docs[d] += s

    return [d for d,_ in sorted(docs.iteritems(), key=lambda x: x[1], reverse=True)][0:topdocs]


def show_most_similar(model, words_docs, df, query):
    pos, neg = parse_pos_neg(query)
    d = most_similar_docs(model, words_docs, pos, neg)
    return df.iloc[d][['title', 'ingredient_txt']]

I AM HERE : try to see if the word2vec search actually worsk.

show_most_similar(m, words_docs, df, 'onion tomato -cilantro')['title']

## -- end of current --

m = models.Word2Vec(['hello','testing','etc'], size=100, window=5, min_count=5, workers=2)


## I AM HERE: get the W rd2Vec to work by inputting the right type: is a a list of strings, or a list of lists, --> chkc the tutorial.s

## approach one: split ingredient text.
w2v = models.Doc2Vec(df['ingredient_txt_no_stopwords'].values, size=100, window=5, min_count=5, workers=4)

m = models.Doc2Vec(df['ingredient_txt_no_stopwords'].values, size=100, window=5, min_count=5, workers=4)

## appproach 2: extract from features matrix.


w2v.vocab

w2v.most_similar(positive=['chicken','thighs'])


df = p.clean_formatting(df)

assert(type(df)) is pd.core.frame.DataFrame, "%r is not a DataFrame." % df
assert(df.shape) == (16526, 7), "Has the wrong shape."

vectorizer, features = p.extract_features(df, title=True)

## trying cosine similarity between topics and search term vector.

m = p.run_model(features, n_topics=40, random_state=0, n_iter=100)

## get topics as words:
topics = p.most_probable_words(m, vectorizer.get_feature_names(), 100)


## word to vec.

df['ingredient_txt'].values[0]




## search string:
raw = 'chicken lemon'

## using gensim similarities.
from gensim import corpora, models, similarities

documents = ["Human machine interface for lab abc computer applications",
              "A survey of user opinion of computer system response time",
              "The EPS user interface management system",
              "System and human system engineering testing of EPS",
              "Relation of user perceived response time to error measurement",
              "The generation of random binary unordered trees",
              "The intersection graph of paths in trees",
              "Graph minors IV Widths of trees and well quasi ordering",
              "Graph minors A survey"]

texts = [[w for w in doc.lower().split()] for doc in documents]

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)

doc = 'human computer interaction'
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow]
print(vec_lsi)

index = similarities.Similarity(lsi[corpus])
