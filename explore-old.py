"""
Script to explore recipe data.
"""

import lda
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import psycopg2
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text




##TODO: make some simple plots: word counts etc.
def feature_counts(vectorizer, dtm):
    "Examine word counts by putting them in an DataFrame."
    wordcounts = pd.DataFrame(np.asarray(dtm.sum(axis=0).T))
    wordcounts.columns = ['count']
    wordcounts['word'] = vectorizer.get_feature_names()

    xx = np.apply_along_axis(lambda x: x > 0, 1, dtm.todense())
    nr_docs = np.sum(xx, 0)
    wordcounts['nr_docs'] = nr_docs
    return wordcounts

def get_stop_words():
    "Combine custom stopwords with standard english ones."
    ingredient_stop_words = [
        'cup', 'tablespoons', 'teaspoons', 'cups', 'tablespoon', 'large',
        'teaspoon', 'inch', 'pound', 'pounds', 'ounces', 'ounce', 'plus',
        'chopped', 'minced', 'cut', 'sliced', 'diced', 'ground', 'grated',
        'peeled', 'fresh', 'lb', 'oz', 'g', 'tbsp', 'tsp', 'f', 'slices',
        'finely', 'thinly', 'medium', 'divided', 'pieces', 'coarse', 'stick',
        'cubes', 'assorted', 'wedges', 'small', 'water', 'white', 'crushed',
        'coarsely', 'temperature', 'room', 'dry', 'packed', 'halved',
        'lengthwise', 'drained', 'powder', 'pale', 'parts', 'lightly',
        'beaten', 'fine', 'plain', 'serving', 'taste', 'removed', 'crumbled',
        'small', 'water', 'white', 'crushed', 'coarsely', 'temperature',
        'room', 'dry', 'packed', 'halved', 'lengthwise', 'drained', 'powder',
        'pale', 'parts', 'lightly', 'beaten', 'fine', 'plain', 'serving',
        'taste', 'removed', 'crumbled', 'trimmed', 'freshly', 'seeded', 'size',
        'reserved', 'garnish', 'quartered', 'discarded', 'mixed', 'torn',
        'bunch', 'stemmed', 'oil', 'salt', 'pepper', 'olive oil', 'garlic',
        'garlic cloves', 'black pepper', 'leaves', 'red', 'olive', 'black',
        'cloves', 'preferably', 'ml'
    ]

    instructions_stop_words = ['minutes']
    return text.ENGLISH_STOP_WORDS.union(
        ingredient_stop_words, instructions_stop_words)



def make_week1_plot(df):


    vectorizer = CountVectorizer(stop_words='english',
                                 ngram_range=(1, 1),
                                 token_pattern='[A-Za-z]+')
    features = vectorizer.fit_transform(df.ingredient_txt)
    ## features is a document x term matrix.

    wc = feature_counts(vectorizer, features)

    ## plot of most common words:
    p1 = wc.sort('count').tail(20).plot('word','count', kind='bar')

    v2 = CountVectorizer(stop_words=get_stop_words(),
                         ngram_range=(1, 1),
                         token_pattern='[A-Za-z]+')
    f2 = v2.fit_transform(df.ingredient_txt)
    ## features is a document x term matrix.

    wc2 = feature_counts(v2, f2)

    ## plot of most common words:
    n = 50

    plt.figure(1)
    plt.subplot(211)
    p1 = wc.sort('count').tail(n).plot('word','count', kind='bar')

    plt.subplot(212)
    p2 = wc2.sort('count').tail(n).plot('word','count', kind='bar')

    plt.tight_layout()
    plt.savefig('fig-word-count-histograms.png')


## extras.
# ## plot of most common words (nr of documents).
# p2 = wc.sort('nr_docs').tail(20).plot('word','nr_docs', kind='bar')

# ## correlation between total count and number of documents:
# np.corrcoef(wc['count'], wc['nr_docs'])


import lda

vectorizer = CountVectorizer(stop_words='english',
                                 ngram_range=(1, 1),
                                 token_pattern='[A-Za-z]+')
features = vectorizer.fit_transform(df.ingredient_txt)
## features is a document x term matrix.

wc = feature_counts(vectorizer, features)


m = lda.LDA(n_topics=40, random_state=0, n_iter=100)
m.fit(features)

## TODO: make convergence plot.

## 1. TODO: extract topic model assignments.

## topic numbers:
m.doc_topic_.argsort(axis=1)
## topic probs:
np.sort(m.doc_topic_, axis=1)

## data frame of:
## document id, topic, probability.

# ## this is for the most probable topic, but we don't need it now.
# x = pd.DataFrame(m.doc_topic_.argsort(axis=1))
# xm = pd.melt(x)
# xm.columns = ['topic','']

x = pd.DataFrame(np.sort(m.doc_topic_, axis=1))
x['doc'] = x.index
xm = pd.melt(x, id_vars=['doc'])
xm.columns = ['doc','topic','prob']

## show distribution of document probs by each topic.

I AM HERE: get this plot of one topic distribution to work, then make a grid.

sns.distplot(xm[xm['topic']==0])



I AM HERE: 1: melt this, to get a 'topic' column from the topic columns
2: merge with df made from np.sort(m.doc_topic_)




np.arange(9), np.sort(m.doc_topic_, axis=1)[20,-10:-1]

## this shows the most probable topics for a specific document.
plt.bar(np.arange(9), np.sort(m.doc_topic_, axis=1)[20,-10:-1])


plt.pcolor(m.doc_topic_)


import seaborn as sns


sns.distplot(m.doc_topic_[:,20])


np.sort(m.doc_topic_, axis=1)[:,20]

sns.distplot(m.doc_topic_[1,:], kde=False)






## 2. TODO: make the website which shows one/three recipes from each

## 3. TODO: process optimization (need frontend to evaluate.)

## 4. TODO: run the topic model using stan.

