"""Functions for exploratory and diagnostic analysis."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def feature_counts(vectorizer, dtm):
    "Examine word counts by putting them in an DataFrame."
    wordcounts = pd.DataFrame(np.asarray(dtm.sum(axis=0).T))
    wordcounts.columns = ['count']
    wordcounts['word'] = vectorizer.get_feature_names()

    xx = np.apply_along_axis(lambda x: x > 0, 1, dtm.todense())
    nr_docs = np.sum(xx, 0)
    wordcounts['nr_docs'] = nr_docs
    return wordcounts


def explore_ingredient_lengths(df):
    print df['ingredient_txt'].str.len().describe()
    plt.figure()
    df['ingredient_txt'].str.len().plot(kind='hist').set_title('Ingredients character count')
    plt.savefig('character-counts.png')

def ingredient_word_count(vectorizer, features):
    wc = feature_counts(vectorizer, features)
    wc.sort('count').tail(25).plot('word','count', kind='bar')
    plt.savefig('word-counts.png')

def plot_loglikelihood_topic(ll):
    "Plot loglikelihoods as a function of number of topics."
    ks = sorted(ll.keys())
    vs = [ll[k] for k in ks]
    plt.plot(ks, vs)

def plot_loglikelihood_iteration(m):
    "Plot loglikelihoods as a function of iteration"
    p = plt.figure()
    plt.plot(m.loglikelihoods_, '-')
    plt.title('Loglikelihood')
    p.savefig('loglikelihood.png')

