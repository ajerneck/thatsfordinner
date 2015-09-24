from app import app
import collections
from flask import render_template, request
import pickle
import psycopg2
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def collapse_topic_words(df):
    "Remove unigrams that appear in bigrams, and collapse bigrams to common words."
    all_topics = collections.defaultdict(list)
    for n, group in df.groupby('label'):
        ks = collections.defaultdict(float)
        for _, row in group.iterrows():
            words = row['word'].split()
            for w in words:
                ks[w] += row['prob']
        all_topics[n] = ', '.join([w.capitalize() for w,i in sorted(ks.items(), key=lambda x: x[1], reverse=True)])
    return all_topics

@app.route('/')
@app.route('/index')
def index():

    with open('app/word_data.pkl') as f:
        print 'reading %s' % f
        word_data = pickle.load(f)
        print 'word_data size'

    with open('app/doc_data.pkl') as f:
        print 'reading %s' % f
        doc_data = pickle.load(f)

    topics = sorted(doc_data.keys(), reverse=True)

    return render_template('compact.html', word_data=word_data, doc_data=doc_data, topics=topics)

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
