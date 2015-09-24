"""Functions for exploratory and diagnostic analysis."""

import matplotlib
import matplotlib.pyplot as plt

def explore_ingredient_lengths(df):
    print df['ingredient_txt'].str.len().describe()
    plt.figure()
    df['ingredient_txt'].str.len().plot(kind='hist').set_title('Ingredients character count')
    plt.savefig('character-counts.png')

def ingredient_word_count(vectorizer, features):
    wc = f.feature_counts(vectorizer, features)
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

