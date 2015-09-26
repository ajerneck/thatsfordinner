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

vectorizer, features = p.extract_features(df, title=False)

ll = {}
for k in range(5, 100, 5):
    print k
    mk  = lda.LDA(n_topics=k, random_state=0, n_iter=200)
    mk.fit(features)
    ll[k] = mk.loglikelihood()


ks = sorted(ll.keys())
vs = [ll[k] for k in ks]
plt.style.use('ggplot')
plt.figure()
plt.plot(ks, vs)
plt.savefig('topics-likelihoods-200-no-title.png')


## save
ll_title = ll
