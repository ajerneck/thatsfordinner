from app import app
import collections
from flask import render_template, request
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

    con = psycopg2.connect(host='localhost', dbname='explore', user='explore', password='Ln2bOYAVCG6utNUSaSZaIVMH')

    with con:
        cur = con.cursor()
        cur.execute("SELECT doc_prob.topic, ingredient_txt, image, url, clean_recipes.title, prob FROM doc_prob, clean_recipes WHERE doc_prob.recipe_key=clean_recipes.key ORDER BY topic desc, rank;")
        query_results = cur.fetchall()
        topics = collections.defaultdict(list)
        for row in query_results:
            topics[row[0]] += [{'topic':row[0], 'ingredient_txt':row[1].decode('ascii', 'ignore'), 'image':row[2], 'url': row[3], 'prob': row[5], 'title':row[4].decode('ascii', 'ignore') }]

    return render_template('index.html', topics=topics)

@app.route('/topics')
def topics():
    con = psycopg2.connect(host='localhost', dbname='explore', user='explore', password='Ln2bOYAVCG6utNUSaSZaIVMH')

    with con:
        cur = con.cursor()


        ## extract most probable words for each topic.
        cur.execute('SELECT * FROM word_probs order by topic, prob desc;')
        word_probs = cur.fetchall()
        word_data = collections.defaultdict(list)
        for row in word_probs:
            word_data[row[2]] += [row[3]]
        for k,v in word_data.items():
            word_data[k] = ', '.join(v)

        ## extract most probable documents for each topic.
        cur.execute("SELECT doc_prob.topic, ingredient_txt, image, url, clean_recipes.title, prob FROM doc_prob, clean_recipes WHERE doc_prob.recipe_key=clean_recipes.key ORDER BY topic, rank;")
        doc_probs = cur.fetchall()

        doc_data = collections.defaultdict(list)
        for row in doc_probs:
            doc_data[row[0]] += [{'ingredient': row[1], 'image':row[2], 'url':row[3], 'title':row[4]}]

        topics = sorted(doc_data.keys(), reverse=True)

    return render_template('topics.html', word_data=word_data, doc_data=doc_data, topics=topics)

@app.route('/compact')
def compact():
    custom = True

    con = psycopg2.connect(host='localhost', dbname='explore', user='explore', password='Ln2bOYAVCG6utNUSaSZaIVMH')

    engine = create_engine("postgresql+psycopg2://explore:Ln2bOYAVCG6utNUSaSZaIVMH@localhost/explore")

    with con:
        cur = con.cursor()

        if custom is True:
            ww = pd.read_sql_table('all_word_probs', engine)
            xx = ww.sort(['label','prob']).groupby('label').tail(10)
            word_data = collapse_topic_words(xx)#

        else:

        ## extract most probable words for each topic.
            cur.execute('SELECT * FROM word_probs order by topic, prob desc;')
            word_probs = cur.fetchall()
            word_data = collections.defaultdict(list)
            for row in word_probs:
                word_data[row[2]] += [row[3]]
                for k,v in word_data.items():
                    cur.execute('SELECT * FROM all_word_probs order by label, prob desc;')
                    word_probs = cur.fetchall()
                    word_data[k] = ', '.join(v)


        ## extract most probable documents for each topic.
        cur.execute("SELECT doc_prob.topic, ingredient_txt, image, url, clean_recipes.title, prob FROM doc_prob, clean_recipes WHERE doc_prob.recipe_key=clean_recipes.key ORDER BY topic, rank;")
        doc_probs = cur.fetchall()

        doc_data = collections.defaultdict(list)
        for row in doc_probs:
            doc_data[row[0]] += [{'ingredient': row[1].decode('ascii', 'ignore'), 'image':row[2], 'url':row[3], 'title':row[4].decode('ascii', 'ignore')}]

        topics = sorted(doc_data.keys(), reverse=True)

    print doc_data
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


@app.route('/all')
def all():
    custom = True

    con = psycopg2.connect(host='localhost', dbname='explore', user='explore', password='Ln2bOYAVCG6utNUSaSZaIVMH')

    engine = create_engine("postgresql+psycopg2://explore:Ln2bOYAVCG6utNUSaSZaIVMH@localhost/explore")

    with con:
        cur = con.cursor()

        if custom is True:
            ww = pd.read_sql_table('all_word_probs', engine)
            xx = ww.sort(['label','prob']).groupby('label').tail(10)
            word_data = collapse_topic_words(xx)#

        else:

        ## extract most probable words for each topic.
            cur.execute('SELECT * FROM word_probs order by topic, prob desc;')
            word_probs = cur.fetchall()
            word_data = collections.defaultdict(list)
            for row in word_probs:
                word_data[row[2]] += [row[3]]
                for k,v in word_data.items():
                    cur.execute('SELECT * FROM all_word_probs order by label, prob desc;')
                    word_probs = cur.fetchall()
                    word_data[k] = ', '.join(v)


        ## extract most probable documents for each topic.
        cur.execute("SELECT doc_prob.topic, ingredient_txt, image, url, clean_recipes.title, prob FROM doc_prob, clean_recipes WHERE doc_prob.recipe_key=clean_recipes.key ORDER BY topic, rank;")
        doc_probs = cur.fetchall()

        doc_data = collections.defaultdict(list)
        for row in doc_probs:
            doc_data[row[0]] += [{'ingredient': row[1].decode('ascii', 'ignore'), 'image':row[2], 'url':row[3], 'title':row[4].decode('ascii', 'ignore')}]

        topics = sorted(doc_data.keys(), reverse=True)

    print topics
    mid_point = len(topics)/2
    print doc_data
    return render_template('all.html', word_data=word_data, doc_data=doc_data, topics=topics, mid_point=mid_point)

