"""
Script to explore recipe data.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import psycopg2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text

matplotlib.style.use('ggplot')



con = psycopg2.connect("host='localhost' dbname='explore' user='explore' password='Ln2bOYAVCG6utNUSaSZaIVMH'")
cursor = con.cursor()

cursor.execute('select category, count(*) from recipes_recipe group by category')

df = pd.read_sql_query('select * from recipes_recipe', con=con)

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


## TODO: run the topic model using the existing code.
I AM HERE: do the topic model, show the results.

THEN: do process optimization over the features and hyperparameters:

    ngrams:
    stop_words
    hi-low feature number range
    topics.






## TODO: run the topic model using stan.

