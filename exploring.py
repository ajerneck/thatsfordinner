"""Script to categorize recipes using topic models."""

## TODO: order
import collections
import lda
import numpy as np
import pandas as pd
import psycopg2
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
sklearn.feature_extraction import text

## TODO: get rid of:
import common

import functions as f


matplotlib.style.use('ggplot')


## TODO: fix unicode here.
df = f.load_data()


# ## Examining and cleaning data


# Extract features
# TODO: remove stopwords before vectorizing, because the stop words don't capture bigrams.

## this is for 

# In[14]:


## this model, with 40 topics, is not bad.
m = lda.LDA(n_topics=40, random_state=0, n_iter=200)
m.fit(features)
print('Finished running model')


# In[38]:

## select best number of topics using loglikelihood.
## it might be that 40 topics is too much.

# # Evaluating the model.
# ## Convergence

# In[15]:


# In[303]:

## assessing stability using k-fold set overlap measures.


r75 = k_fold_lda(n_topics=75, random_state=0, n_iter=300)
r40 = k_fold_lda(n_topics=40, random_state=0, n_iter=300)


# results = [lda.LDA(n_topics=best_k, random_state=0, n_iter=100).fit(features[i

    


# In[314]:

########################################
# Extract and compare set memberships. #
########################################

def set_memberships(r):
    map(lambda x: np.argmax(x.doc_topic_, axis=1, r)

for i = range(0, 5)
    for j = range(1, 4)
    diff = np.argmax(r4[i].doc_topic_, axis=1)


# In[15]:

## assessing stability using two random subsets..
n = features.shape[0]
size = int(n * 0.8)

i0 = np.random.random_integers(0, n, size)
i1 = np.random.random_integers(0, n, size)

f0 = features[i0, :]
f1 = features[i1, :]

print('running model on sample 0...')
m0 = lda.LDA(n_topics=40, random_state=0, n_iter=100)
m0.fit(f0)

print('running model on sample 1...')
m1 = lda.LDA(n_topics=40, random_state=0, n_iter=100)
m1.fit(f1)

print('Finished running models')


# In[16]:

a0 = zip(i0, np.argmax(m0.doc_topic_, axis=1))
a1 = zip(i1, np.argmax(m0.doc_topic_, axis=1))

## make 

t0 = collections.defaultdict(set)
for doc, topic in a0:
    t0[topic].add(doc)

t1 = collections.defaultdict(set)
for doc, topic in a1:
    t1[topic].add(doc)


# In[17]:

## filter out elements not in both sets.
canon = set(i0).union(set(i1))
for k,v in t0.iteritems():
    t0[k] = t0[k].intersection(canon)
for k,v in t1.iteritems():
    t1[k] = t1[k].intersection(canon)

## compare set assignments.


# In[18]:

def diff(t0, t1):
    """Calculate set membership differences as a stability measure.
    """


    diff = 0
    for k0, x0 in t0.iteritems():
        # print 'topic:', k0
        d = [len(x0.difference(x1))/float(len(x0.union(x1))) for x1 in t1.values()]
        # print 'differences:', d
        # print 'min diff:', min(d)
        # print 'current diff', diff
        diff += min(d)
    return diff


# In[19]:

print diff(t0, t1)


# # Assessing topics

# In[16]:


## Extracting topic data.
## most probable words by topic.
## TODO: check if these are properly sorted within each topic.
w = f.most_probable_words(m, vectorizer.get_feature_names(), 10)
w.columns = ['rank','topic','word','prob']

## most probable documents by topic.
# np.apply_along_axis(lambda i: df.iloc[i]['title'], 1, doc_ids)
doc_ids = np.argsort(m.doc_topic_, axis=0)[-4:-1,:].T
doc_probs = np.sort(m.doc_topic_, axis=0)[-4:-1,:].T


# In[986]:


f.show_topics(m, df, doc_probs, doc_ids, w)


# In[971]:

# Plotting word distributions for each topic.
wb = f.most_probable_words(m, vectorizer.get_feature_names(), 10)
wb.columns = ['rank','topic','word','prob']

## make figure of word distributions for each topic.
g = sns.FacetGrid(wb, col='topic', col_wrap=10)
p = g.map(sns.barplot, 'word', 'prob')
## save figure for easier viewing.
p.savefig('word_dist_by_topic.png')

## TODO: figure out way of examining probs of words in relation to topic coherence:
## high average prob?

## make figure of document distributions for each topic.
## for each topic, show distribution of documents.





# In[ ]:

## examine topics:
## 14: one very probable word.
## 32: many very probable words.
## 30: no very probable words.




# In[18]:

## TODO: store one set of results for each run.

con = f.make_engine()

## massage document ids and probabilities into form suitable for database.
di = pd.DataFrame(doc_ids)
di['topic'] = di.index
di = pd.melt(di, id_vars='topic')
di.columns = ['topic','rank','recipe_key']

dp = pd.DataFrame(doc_probs)
dp['topic'] = dp.index
dp = pd.melt(dp, id_vars='topic')
dp.columns = ['topic','rank','prob']

dd = pd.merge(di, dp)
dd.to_sql('doc_prob', con, if_exists='replace')

# store recipes
df['key'] = df.index
## assign the most probable topic to each recipe.
df['topic'] = np.argmax(m.doc_topic_, axis=1)
df.to_sql('clean_recipes', con, if_exists='replace', index=False)

# store words
w.columns = ['rank','topic','word','prob']
w.to_sql('word_probs', con, if_exists='replace')


# In[285]:


xx = pd.merge(df, dd, left_on='key', right_on='recipe_key', how='right')

## topics with low word probs.
## but, they seem pretty good.
print 'topics with low word probs.'
for n, g in xx[xx['topic'].isin([5,8,16,18,30])].groupby('topic'):
    print 'topic: %s' % n
    print g[['title','prob']].sort('prob').to_string()

##
print '='*80
print 'topics with one high word prob.'
print '='*80
for n, g in xx[xx['topic'].isin([1,4,9,14,21])].groupby('topic'):
    print 'topic: %s' % n
    print g[['title','prob']].sort('prob').to_string()




# In[469]:

## relationship between doc prob and length:
dpa = pd.DataFrame({'max_prob':np.max(m.doc_topic_, axis=1), 'topic':np.argmax(m.doc_topic_, axis=1)})
dpa = df.join(dpa)
dpa['ingredient_len'] = dpa['ingredient_txt'].str.len()

dpa.plot('ingredient_len', 'mdpa_prob', kind='scatter')

# g = sns.FacetGrid(dpa, col='topic', col_wrap=10)
# p = g.map(sns.barplot, 'word', 'prob')
# ## save figure for easier viewing.
# p.savefig('word_dist_by_topic.png')



# In[39]:

## generating recipes.
reload(f)
ww = f.all_word_probs(m, vectorizer.get_feature_names())
ww.to_sql('all_word_probs', con, if_exists='replace')


# In[40]:

for n, g in ww.groupby('label'):
    print n
    print [', '.join(np.random.choice(g['word'], size=5, p=g['prob'])) for _ in range(0,1)]
    

