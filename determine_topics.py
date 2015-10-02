## select best number of topics using loglikelihood.

import collections
import itertools
import matplotlib.pyplot as plt

import numpy as np
import lda
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

vectorizer, features = p.extract_features(df, title=True)

ll = {}
ms = []
for k in range(5, 100, 5):
    print k
    mk  = lda.LDA(n_topics=k, random_state=0, n_iter=1000)
    mk.fit(features)
    ll[k] = mk.loglikelihood()
    ms.append(mk)

ll_5_100_1000 = ll
ms_5_100_1000 = ms

plot_lls(ll_5_100_1000, 'Loglikelihood by number of topics', 'll-topics-5-100-1000.png')

def plot_lls(ll, title, filename):
    ks = sorted(ll.keys())
    vs = [ll[k] for k in ks]
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(ks, vs)
    plt.title(title)
    plt.savefig(filename)


x1 = pd.DataFrame(ll_5_100_200.items(), columns=['topics','ll'])
x1['run'] = 1
x2 = pd.DataFrame(ll_20_40_400.items(), columns=['topics','ll'])
x2['run'] = 2
x3 = pd.DataFrame(ll_5_100_1000.items(), columns=['topics','ll'])
x3['run'] = 3

xx = pd.concat([x1, x2, x3])

xx = xx.sort('topics')
xx.index = xx['topics']

plt.figure()
colors = ['red','green','blue']
for n, g in xx.groupby('run'):
    plt.plot(g['topics'], g['ll'], color=colors[n-1])
plt.savefig('testing.png')

plt.figure()
g = xx[xx['run']==3]
g['ll'] = g['ll'] / 10000.0
plt.plot(g['topics'], g['ll'], color=colors[n-1])
plt.xlabel('Number of topics')
plt.ylabel('Model fit (loglikelihood)')
plt.savefig('loglikelihood-topics.png')



## I AM HERE: do a nice plot for the demo.



xx.plot('x=')


## save
ll_title = ll
