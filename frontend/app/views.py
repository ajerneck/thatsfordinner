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
        cur.execute("SELECT topic, ingredient_txt, image, url, clean_recipes.title, prob FROM doc_prob, clean_recipes WHERE doc_prob.recipe_key=clean_recipes.key ORDER BY topic, rank;")
        query_results = cur.fetchall()
        topics = collections.defaultdict(list)
        for row in query_results:
            topics[row[0]] += [{'topi':row[0], 'ingredient_txt':row[1].decode('ascii', 'ignore'), 'image':row[2], 'url': row[3], 'prob': row[5], 'title':row[4].decode('ascii', 'ignore') }]

    return render_template('index.html', topics=topics)
