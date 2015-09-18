from app import app
import collections
from flask import render_template, request
import psycopg2

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

    return render_template('compact.html', word_data=word_data, doc_data=doc_data, topics=topics)


