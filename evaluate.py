"""Different ways of evaluating the topic models."""

import collections
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import psycopg2
import seaborn as sns

import explore as e
import production as p

df = p.load_data()
df = p.clean_formatting(df)
df = p.remove_stopwords(df)

assert(type(df)) is pd.core.frame.DataFrame, "%r is not a DataFrame." % df
assert(df.shape) == (16526, 7), "Has the wrong shape."

vectorizer, features = p.extract_features(df, title=True)

## VERSION 1:
## run model 2 times on different 80% of the data.

def run_on_sample(features, prop, **kwargs):

    n = features.shape[0]
    ix = np.random.choice(n, size=int(n*prop), replace=False)

    m = p.run_model(features[ix,:], random_state=0, **kwargs)

    s = collections.defaultdict(set)
    for k,v in zip(np.argmax(m.doc_topic_, axis=1), ix):
        s[k].add(v)
    return s


def select_common_one(s, both):
    for k,v in s.iteritems():
        s[k] = v.intersection(both)
    return s

def select_common(sa, sb):
    a =  set(itertools.chain(*sa.values()))
    b = set(itertools.chain(*sb.values()))
    both = a.intersection(b)

    return (select_common_one(sa, both), select_common_one(sb, both))

def sim(s0, s1):

    dim = len(s0.keys())

    all_diffs = np.zeros((dim, dim))
    for ik, iv in s0.iteritems():
        for jk, jv in s1.iteritems():
            x =  len(iv.intersection(jv)) / float(len(iv.union(jv)))
            all_diffs[ik, jk] = x

    print 'Overlap score: %.2f' % (np.mean(np.max(all_diffs, axis=1)))
    print 'Overlap score: %.2f' % (np.mean((all_diffs)) * dim)
    return all_diffs



s40_50_a = run_on_sample(features, 0.50, n_topics=40, n_iter=10)
s40_50_b = run_on_sample(features, 0.50, n_topics=40, n_iter=10)

s40_50_a, s40_50_b = select_common(s40_50_a, s40_50_b)

a = sim(s40_50_a, s40_50_b)
sns.heatmap(a[:, np.argmax(a, axis=1)])
plt.savefig('overlaps-sorted-40-50.png')

s75_50_a = run_on_sample(features, 0.50, n_topics=75, n_iter=100)
s75_50_b = run_on_sample(features, 0.50, n_topics=75, n_iter=100)

s75_50_a, s75_50_b = select_common(s75_50_a, s75_50_b)

a = sim(s75_50_a, s75_50_b)
sns.heatmap(a[:, np.argmax(a, axis=1)])
plt.savefig('overlaps-sorted-75-50.png')




## complete overlap

t0 = {0: set(range(0, 10)), 1: set(range(20,30)), 2: set((40,50))}
t1 = {0: set(range(0, 9) + [20]), 2: set(range(21,29) + [10]), 1: set((40,50))}
a = sim(t0, t1)
sns.heatmap(a[:, np.argmax(a, axis=1)])
plt.savefig('overlaps-test-sorted.png')




## random assignment:
x = range(0, len(both))
np.random.shuffle(x)

set_sizes = 







##
ss0 = {}
for k in np.argmax(all_diffs)


sns.heatmap(all_diffs)
plt.savefig('overlaps.png')



## get the smallest set difference between 2:



results = []
for i in range(0, 1):
    ix = np.random.choice(n, size=int(n*0.8), replace=False)
    x = features[ix,:]
    m = p.run_model(x, n_topics=75, random_state=0, n_iter=4)
    results.append({'model': m, 'topics': np.argax(m.doc_topic_, axis=1)})


## extract set memberships (only need to reorder by minimum distance)

s0 = np.argmax(models[0].doc_topic_, axis=1)
s1 = np.argmax(models[1].doc_topic_, axis=1)

ss0 = collections.defaultdict(set)
for v, k in enumerate(s0):
    ss0[k].add(v)

ss1 = collections.defaultdict(set)
for v, k in enumerate(s1):
    ss0[k].add(v)


for ik,iv in s0.iteritems():
    for jk, jv in s1.iteritems():
        print min([iv ^ jv])



def k_f ld_run(**kwargs):
    "Run 5 lda models on 20% of the data each."
    results = []
    j = 0
    for i in map(int, np.linspace(0, features.shape[0], 6)[1:]):
        results.append(lda.LDA(**kwargs).fit(features[j:i-1,:]))
        j = i
    return results

## calculate the

## VERSION 2:
