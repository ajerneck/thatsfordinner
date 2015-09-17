import numpy as np
import pandas as pd
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

    ww.columns = ['rank', 'label', 'word', 'prob']
    return ww



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
        'cloves', 'preferably', 'ml', 'shredded','dried', 'g', 'pieces', 'inch', 'cut', 'size','bite', 'pinch','clove','taste', 'large', 'grated', 'half' , 'minced' , 'peeled' , 'seeded' , 'shredded', 'dried','piece','for','inch', 'cubed']
    return text.ENGLISH_STOP_WORDS.union(ingredient_stop_words)

def show_topics(m, df, doc_probs, doc_ids, w):
    print('='*70)
    for t in range(m.n_topics):
        print('topic: %s' % t)
        print('documents:')
        print pd.DataFrame([df.iloc[doc_ids[t,:]]['title'].values, doc_probs[t,:]]).T.sort(1, ascending=False).to_string(header=False, index=False)
        #    print('\n'.join(df.iloc[doc_ids[t,:]]['title']))
        print('-----'.join(df.iloc[doc_ids[t,:]]['ingredient_txt_no_stopwords']))
        print('-'*70)
        print w[w['topic']==t][['word','prob']].sort('prob', ascending=False).T.to_string(index=False, header=False, float_format=lambda x: '% 4.3f' % x)
        print('='*70)

