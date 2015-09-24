"Script to run the entire analysis."

import collections
import itertools
import numpy as np
import pandas as pd
import pickle
import psycopg2

import explore as e
import production as p

df = p.load_data()
df = p.clean_formatting(df)
df = p.remove_stopwords(df)

assert(type(df)) is pd.core.frame.DataFrame, "%r is not a DataFrame." % df
assert(df.shape) == (16526, 7), "Has the wrong shape."

vectorizer, features = p.extract_features(df)

## run model.
m = p.run_model(features, n_topics=75, random_state=0, n_iter=100)



## extract and prepare most probable documents.

def save_data_for_frontend(model, vectorizer, df):

    doc_ids = np.argsort(model.doc_topic_, axis=0)[-4:-1,:].T
    doc_probs = np.sort(model.doc_topic_, axis=0)[-4:-1,:].T
    topic_total_probs = np.sum(doc_probs, axis=1)
 
    ## extract and prepare most probable words.
    ## split bigrams and take the unique set of the resulting word list.
    w = p.most_probable_words(model, vectorizer.get_feature_names(), 10)
    word_data = collections.defaultdict(list)
    for topic, g in w.groupby('topic'):
        word_data[topic] = ', '.join([w.capitalize() for w in p.unique(itertools.chain(*g.sort('prob', ascending=False)['word'].str.split(' ').values))])
        # word_data[topic] = ', '.join([str(g['prob'].sum())] + [w.capitalize() for w in p.unique(itertools.chain(*g.sort('prob', ascending=False)['word'].str.split(' ').values))])
    # for k,v in word_data.iteritems():
    #     print k
    #     print topic_total_probs[k]
    #     word_data[k] = v + str(topic_total_probs[k])


    with open('frontend/app/word_data.pkl', 'w') as f:
        pickle.dump(word_data, f)


    di = pd.DataFrame(doc_ids)
    di['topic'] = di.index
    di = pd.melt(di, id_vars='topic')
    di.columns = ['topic','rank','key']
    dp = pd.DataFrame(doc_probs)
    dp['topic'] = dp.index
    dp = pd.melt(dp, id_vars='topic')
    dp.columns = ['topic','rank','prob']

    dd = pd.merge(di, dp)

    ## merge in document data for the most probable documents.
    df['topic'] = np.argmax(model.doc_topic_, axis=1).T
    df['key'] = df.index
    most_probable_docs = pd.merge(df, dd)
    ## TODO: do the decoding here.

    doc_data = collections.defaultdict(list)
    for topic, g in most_probable_docs.groupby('topic'):
        row = g.sort('prob')[['ingredient_txt','image','url','title']].values
        doc_data[topic] = map(lambda x: dict(zip(['ingredient','image','url','title'], x)), row)
    with open('frontend/app/doc_data.pkl', 'w') as f:
        pickle.dump(doc_data, f)


