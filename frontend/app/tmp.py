import collections
import psycopg2

con = psycopg2.connect(host='localhost', dbname='explore', user='explore', password='Ln2bOYAVCG6utNUSaSZaIVMH')

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

topics = doc_data.keys()




## TODO:
    # extract max topic assignment for each topic
    ## cur.execute('SELECT topic, ingredient_txt, image, url, title')
    ## topic_docs = 



topic_data = collections.defaultdict(dict)

