"Script to run the entire analysis."

import collections
import itertools
import numpy as np
import pandas as pd
import pickle
import psycopg2
from sklearn.feature_extraction.text import CountVectorizer

import explore as e
import production as p

df = p.load_data()
df = p.clean_formatting(df)
df = p.remove_stopwords(df)

assert(type(df)) is pd.core.frame.DataFrame, "%r is not a DataFrame." % df
assert(df.shape) == (16526, 7), "Has the wrong shape."

vectorizer, features = p.extract_features(df, title=True)

## run model.
m = p.run_model(features, n_topics=45, random_state=0, n_iter=100)



## extract and prepare most probable documents.

def save_data_for_frontend(model, vectorizer, df):

    doc_ids = np.argsort(model.doc_topic_, axis=0)[-5:-1,:].T
    doc_probs = np.sort(model.doc_topic_, axis=0)[-5:-1,:].T
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
    df['topic_prob'] = np.max(model.doc_topic_, axis=1).T
    df['key'] = df.index
    most_probable_docs = pd.merge(df, dd)
    ## TODO: do the decoding here.

    doc_data = collections.defaultdict(list)
    for topic, g in most_probable_docs.groupby('topic'):
        row = g.sort('prob')[['ingredient_txt','image','url','title', 'key']].values
        doc_data[topic] = map(lambda x: dict(zip(['ingredient','image','url','title','key'], x)), row)
    with open('frontend/app/doc_data.pkl', 'w') as f:
        pickle.dump(doc_data, f)

    engine = p.make_engine()
    df.to_sql('clean_recipes', engine, if_exists='replace')

save_data_for_frontend(m, vectorizer, df)

## calculate and save cosine similarities for standard searching.
## prepare beforehand.
vv = CountVectorizer(
    stop_words=p.get_stop_words()
    , ngram_range=(1, 1)
    , token_pattern = '[A-Za-z]+'
)

search_cols = df['ingredient_txt_no_stopwords'].str.cat(df['title'].values, sep=' ')
vv = vv.fit(search_cols)
all_features = vv.transform(search_cols)

with open('frontend/app/search_vectorizer.pkl', 'w') as f:
    pickle.dump(vv, f)
with open('frontend/app/search_all_features.pkl', 'w') as f:
    pickle.dump(all_features, f)
