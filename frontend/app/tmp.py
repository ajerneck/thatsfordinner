import collections
import psycopg2
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql+psycopg2://explore:Ln2bOYAVCG6utNUSaSZaIVMH@localhost/explore")
#con = psycopg2.connect(host='localhost', dbname='explore', user='explore', password='Ln2bOYAVCG6utNUSaSZaIVMH')

ww = pd.read_sql_table('all_word_probs', engine)

xx = ww.sort(['label','prob']).groupby('label').tail(20)
## get the 10 most probable within each group.

ws = xx[xx['label']==0]['word'].values

## loop over words, filter out those that are in other bigsams
wss = [w.split() for w in ws]
bs = []
us = []
for w in wss:
    if len(w) > 1:
        bs.append(w)
    else:
        us.append(w)

keep = []
for u in us:
    hits = [b for b in bs if u[0] in b]
    if hits == []:
        keep.append(u)

## this works.
keep = []
for w in itertools.chain(*wss):
    print w
    if w not in keep:
        keep.append(w)

## TODO: sum up the probabilities:



## do nlp on words to identify nouns.


## do decorate, sort, undecorate



for row in ws.values.tolist():
    print row
    words = row[0].split()
    prob = row[1]
    for w in words:
        ks[w] += prob

k = collections.defaultdict(float)
for w in itertools(*wss):
    k[w] += 


k = dict.fromkeys(itertools.chain(*wss), 0)






x = {}
import itertools
dict.fromkeys(itertools.chain(*wss))

ws = xx[xx['label']==0][['word','prob']]

ws['word'].values
ws['word'].values

wss





bs = [w for w in wss if len(w)>1]

for w in ws:
    if 

use 'low' in 'low chicken'



# cur = con.cursor()
# cur.execute('select * from all_word_probs order by label')
# ww = cur.fetchall()

gen_recipes = collections.defaultdict(list)
for topic, group in ww.groupby('label'):
    # print topic
    # print [(np.random.choice(group['word'], size=5, p=group['prob'])) for _ in range(0,2)]
    x = [np.random.choice(group['word'], size=5, p=group['prob']).tolist() for i in range(0, 2)]
    gen_recipes[topic] += [map(lambda i: ', '.join(i), x)]

print gen_recipes


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

topics = doc_data.keys()




## TODO:
    # extract max topic assignment for each topic
    ## cur.execute('SELECT topic, ingredient_txt, image, url, title')
    ## topic_docs = 



topic_data = collections.defaultdict(dict)

