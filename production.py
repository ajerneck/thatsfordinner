"""Functions for the production analysis."""

import lda
import numpy as np
import pandas as pd
import psycopg2
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sqlalchemy import create_engine


def make_engine():
    "Return a sqlalchemy engine connection to explore database."
    return create_engine("postgresql+psycopg2://explore:Ln2bOYAVCG6utNUSaSZaIVMH@localhost/explore")


def load_epicurious():
    con = make_engine()

    df_ep = pd.read_sql_table('recipes_recipe', con)
    df_ep = df_ep[['title','ingredient_txt','url','image']]
    df_ep['source'] = 'epicurious.com'

    print('Loaded %s records from epicurious.com' % df_ep.shape[0])
    return df_ep

def load_allrecipes():
    con = make_engine()

    df_ar = pd.read_sql_table('allrecipes', con)
    df_ar = df_ar[['data-name','ingredients','url','data-imageurl']]
    df_ar.columns = ['title','ingredient_txt','url','image']
    df_ar['source'] = 'allrecipes.com'
    df_ar = df_ar.drop_duplicates('url')
    df_ar.reset_index()

    print('Loaded %s records from allrecipes.com' % df_ar.shape[0])
    return df_ar

def load_data():

    df_ep = load_epicurious()
    df_ar = load_allrecipes()
    df = pd.concat([df_ep, df_ar], ignore_index=True)

    print('Loaded %s records in total' % df.shape[0])
    return df

def clean_formatting(df):
    ## the number 20 comes from examining the character length histograms:
    df = df[df['ingredient_txt'].str.len() > 20]
    df = df.reset_index()
    ## clean up quoting.
    df['title'] = df['title'].str.replace('\\\u0027', "'")
    pattern = "[\"\']"
    for k in ['title', 'ingredient_txt', 'url', 'image']:
        df[k] = df[k].str.replace(pattern, '')
        ## formatting ingredients.
        ## df['ingredient_txt'] = df['ingredient_txt'].str.replace('\n',' ')

    return df

def get_stop_words():
    "Combine custom stopwords with standard english ones."
    ingredient_stop_words = [ 'cup', 'tablespoons', 'teaspoons', 'cups',
        'tablespoon', 'large', 'teaspoon', 'inch', 'pound', 'pounds', 'ounces',
        'ounce', 'plus', 'chopped', 'minced', 'cut', 'sliced', 'diced',
        'ground', 'grated', 'peeled', 'fresh', 'lb', 'oz', 'g', 'tbsp', 'tsp',
        'f', 'slices', 'finely', 'thinly', 'medium', 'divided', 'pieces',
        'coarse', 'stick', 'cubes', 'assorted', 'wedges', 'small', 'water',
        'white', 'crushed', 'coarsely', 'temperature', 'room', 'dry', 'packed',
        'halved', 'lengthwise', 'drained', 'powder', 'pale', 'parts', 'lightly',
        'beaten', 'fine', 'plain', 'serving', 'taste', 'removed', 'crumbled',
        'small', 'water', 'white', 'crushed', 'coarsely', 'temperature', 'room',
        'dry', 'packed', 'halved', 'lengthwise', 'drained', 'powder', 'pale',
        'parts', 'lightly', 'beaten', 'fine', 'plain', 'serving', 'taste',
        'removed', 'crumbled', 'trimmed', 'freshly', 'seeded', 'size',
        'reserved', 'garnish', 'quartered', 'discarded', 'mixed', 'torn',
        'bunch', 'stemmed', 'oil', 'salt', 'pepper', 'olive oil', 'garlic',
        'garlic cloves', 'black pepper', 'leaves', 'red', 'olive', 'black',
        'cloves', 'preferably', 'ml', 'shredded','dried', 'g', 'pieces', 'inch',
        'cut', 'size','bite', 'pinch','clove','taste', 'large', 'grated', 'half'
        , 'minced' , 'peeled' , 'seeded' , 'shredded',
        'dried','piece','for','inch', 'cubed', 'kosher', 'seed', 'juice',
        'strip', 'strips', 'water' ]

    return text.ENGLISH_STOP_WORDS.union(ingredient_stop_words)

def remove_stopwords(df):
    """Remove stopwords, taking punctuation into account.

    We need to use this function because CountVectorizer
    does not handle punctuation.
    """
    def rm_stopwords(stopwords, x):
        return ' '.join([w for w in x.split() if w.strip() not in stopwords])

    ## replace punctuation to improve tokenizing and stop word filtering.
    df['ingredient_txt_no_stopwords'] = df['ingredient_txt'].str.replace('[\W]', ' ')
    df['ingredient_txt_no_stopwords'] = map(lambda x: rm_stopwords(get_stop_words(), x), df['ingredient_txt_no_stopwords'])
    return df

def extract_features(df, title):
    "Extract features from ingredients using CountVectorizer."

    vectorizer = CountVectorizer(
        stop_words=get_stop_words()
        , ngram_range=(1, 1)
        , token_pattern='[A-Za-z]+'
        , min_df = 10
        , max_df = 0.25
    )
    if title:

        features = vectorizer.fit_transform(df['ingredient_txt_no_stopwords'].str.cat(df['title'].values, sep=' '))
    else:
        features = vectorizer.fit_transform(df['ingredient_txt_no_stopwords'])

    return (vectorizer, features)

def run_model(features, **model_parameters):
    "Run lda model on features matrix, using model_parameters"
    m = lda.LDA(**model_parameters)
    return m.fit(features)

def select_best_nr_topics(features, start, stop, step):
    """Select the number of topics, defined by the range start, stop, step,
    that maximizes the loglikelihood.
    """
    ll = {}
    for k in range(5, 200, 5):
        print k
        mk  = lda.LDA(n_topics=k, random_state=0, n_iter=400)
        mk.fit(features)
        ll[k] = mk.loglikelihood()

def k_fold_run(**kwargs):
    "Run 5 lda models on 20% of the data each."
    results = []
    j = 0
    for i in map(int, np.linspace(0, features.shape[0], 6)[1:]):
        results.append(lda.LDA(**kwargs).fit(features[j:i-1,:]))
        j = i
    return results


def most_probable_words(model, vocabulary, num_words):
    """
    Return a DataFrame of the most probable words for each topic,
    given a model, vocabulary, and number of words.
    """
    ## create array of vocabulary, sorted by topic
    ## probabilities, one row for each topic.
    vocab = np.asarray(vocabulary)[np.argsort(model.topic_word_)]
    wp = np.sort(model.topic_word_)

    ## select n most probable words, which are the right-most
    ## columns in the vocab array.
    words = vocab[:, -num_words:-1]

    words = pd.DataFrame(words.T)
    words['rank'] = words.index
    words = pd.melt(words, id_vars='rank')

    word_probs = wp[:, -num_words:-1]
    word_probs = pd.DataFrame(word_probs.T)
    word_probs['rank'] = word_probs.index
    word_probs = pd.melt(word_probs, id_vars='rank')

    ww = words.merge(word_probs, on=['rank', 'variable'])

    ww.columns = ['rank', 'topic', 'word', 'prob']
    return ww



def feature_counts(vectorizer, dtm):
    "Examine word counts by putting them in an DataFrame."
    wordcounts = pd.DataFrame(np.asarray(dtm.sum(axis=0).T))
    wordcounts.columns = ['count']
    wordcounts['word'] = vectorizer.get_feature_names()

    xx = np.apply_along_axis(lambda x: x > 0, 1, dtm.todense())
    nr_docs = np.sum(xx, 0)
    wordcounts['nr_docs'] = nr_docs
    return wordcounts

def unique(seq):
    seen = set()
    seen_add = seen.add
    return [ x for x in seq if not (x in seen or seen_add(x))]




