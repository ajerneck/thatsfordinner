from app import app
import collections
from flask import render_template, request
import pickle
import psycopg2
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.metrics.pairwise import cosine_similarity

@app.route('/')
@app.route('/index')
def index():

    print '-'*10, 'starting', '-'*10

    with open('app/word_data.pkl') as f:
        print 'reading %s' % f
        word_data = pickle.load(f)
        print 'word_data size'

    with open('app/doc_data.pkl') as f:
        print 'reading %s' % f
        doc_data = pickle.load(f)

    with open('app/search_vectorizer.pkl') as f:
        vectorizer = pickle.load(f)
    with open('app/search_all_features.pkl') as f:
        all_features = pickle.load(f)

    sq = request.args.get('sq')
    if sq is not None:
        results = cosine_search(sq, vectorizer, all_features, 500)
        ## filter on search query by taking the 500 recipes with the closest non-zero cosine similarity to the query string.
        print doc_data.keys()
        ## filter out recipes that are not in the search result kesy.
        to_rm = []
        for k,v in doc_data.iteritems():
            x = [r for r in v if r['key'] in results]
            if x != []:
                doc_data[k] = x
            else:
                to_rm.append(k)
        ## filter out topics with no results.
        for k in to_rm:
            del(doc_data[k])


    topics = sorted(doc_data.keys(), reverse=True)

    return render_template('index.html', word_data=word_data, doc_data=doc_data, topics=topics)

def cosine_search(query, vectorizer, features, n):
    print 'query string: ', query
    query_features = vectorizer.transform([query])
    sims = cosine_similarity(features, query_features).ravel()
    results = np.argsort(sims)
    results = results[np.nonzero(np.sort(results))]
    results = results[-n:-1]
    scores = np.sort(sims)[-n:-1]
    print results
    print scores
    return results


def decode_string(s):
    if type(s) is str:
        return s.decode('ascii', errors='ignore')
    else:
        return s


@app.route('/all/<topic>/')
def all(topic):
    print '-'*30, 'starting', '-'*30
    sq = request.args.get('sq')
    print sq
    con = psycopg2.connect(host='localhost', dbname='explore', user='explore', password='Ln2bOYAVCG6utNUSaSZaIVMH')
    cursor = con.cursor()
    print topic

    ## I AM HERE: do the cosine search agains sq, then only return those elements.


    with cursor:
        cursor.execute('SELECT * FROM clean_recipes where topic=%s order by topic_prob desc;' % topic)
        results =  cursor.fetchall()
        ## decode strings
        for i,row in enumerate(results):
            results[i] = map(decode_string, row)

    return render_template('all.html', results=results)


@app.route('/lucky')
def lucky():

    engine = create_engine("postgresql+psycopg2://explore:Ln2bOYAVCG6utNUSaSZaIVMH@localhost/explore")

    ww = pd.read_sql_table('all_word_probs', engine)

    gen_recipes = collections.defaultdict(list)

    for topic, group in ww.groupby('label'):
        x = [np.random.choice(group['word'], size=8, p=group['prob'], replace=False).tolist() for i in range(0, 3)]
        gen_recipes[topic] += map(lambda i: ', '.join(i), x)


    print gen_recipes[39]
    return render_template('lucky.html', recipes=gen_recipes)
