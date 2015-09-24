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

def create_sample(features, prop):
    n = features.shape[0]
    ix = np.random.choice(n, size=int(n*prop), replace=False)
    return ix

def run_on_sample(features, ix, **kwargs):
    return p.run_model(features[ix,:], random_state=0, **kwargs)

def get_topic_assignments(ix, m):
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
            print (ik, jk, x)

    print 'Overlap score: %.2f' % (np.mean(np.max(all_diffs, axis=1)))
    print 'Overlap score: %.2f' % (np.mean((all_diffs)) * dim)
    return all_diffs

def sim2(s0, s1):

    dim = len(s0.keys())

    all_diffs = np.zeros(dim)
    for ik, iv in s0.iteritems():
        all_diffs[ik] = np.max([len(iv.intersection(jv))/float(len(iv.union(jv))) for jk, jv in s1.iteritems()])

        # for jk, jv in s1.iteritems():
        #     x =  len(iv.intersection(jv)) / float(len(iv.union(jv)))
        #     all_diffs[ik, jk] = x
        #     print (ik, jk, x)

    # print 'Overlap score: %.2f' % (np.mean(np.max(all_diffs, axis=1)))
    # print 'Overlap score: %.2f' % (np.mean((all_diffs)) * dim)
    return all_diffs

## word assignments.



## topic assignments.

s40_50_a = get_topic_assignments(run_on_sample(features, 0.50, n_topics=40, n_iter=10))
s40_50_b = get_topic_assignments(run_on_sample(features, 0.50, n_topics=40, n_iter=10))

s40_50_a, s40_50_b = select_common(s40_50_a, s40_50_b)

a = sim(s40_50_a, s40_50_b)
sns.heatmap(a[:, np.argmax(a, axis=1)])
plt.savefig('overlaps-sorted-40-50.png')

s75_50_a = get_topic_assignments(run_on_sample(features, 0.50, n_topics=75, n_iter=100))
s75_50_b = get_topic_assignments(run_on_sample(features, 0.50, n_topics=75, n_iter=100))

s75_50_a, s75_50_b = select_common(s75_50_a, s75_50_b)

a = sim(s75_50_a, s75_50_b)
sns.heatmap(a[:, np.argmax(a, axis=1)])
plt.savefig('overlaps-sorted-75-50.png')
## TODO: random assignments to get baseline.
## get the topic sizes, then assign the ids to these topics randomly.


## complete overlap

t0 = {0: set(range(0, 10)), 1: set(range(20,30)), 2: set((40,50))}
t1 = {0: set(range(0, 9) + [20]), 2: set(range(21,29) + [10]), 1: set((40,50))}
a = sim(t0, t1)
a2 = sim2(t0, t1)

sns.heatmap(a[:, np.argmax(a, axis=1)])
plt.savefig('overlaps-test-sorted.png')
t0 = {0: set([0, 1]), 2: set([2,3]), 1: set([4,5])}
t1 = {0: set([0, 1]), 2: set([4,5]), 1: set([2,3])}
a = sim(t0, t1)
a2 = sim2(t0, t1)


ix_a = create_sample(features, 0.8)
ix_b = create_sample(features, 0.8)
m_a = get_topic_assignments(ix_a, run_on_sample(features, ix_a, n_topics=20, n_iter=100))
m_b = get_topic_assignments(ix_b, run_on_sample(features, ix_a, n_topics=20, n_iter=100))

m_a, m_b = select_common(m_a, m_b)
a = sim(m_a, m_b)
ax = pd.DataFrame(a)
ax.index = np.argmax(a, axis=0)
ax = ax.sort()




##I AM HERE: compare the topic words instead -- should be better?

def get_word_assignments(vectorizer, m, n):
    x = dict(enumerate(np.asarray(vectorizer.get_feature_names())[np.argsort(m.topic_word_, axis=1)[:,-n:-1]]))
    for k,v in x.iteritems():
        x[k] = set(v)
    return x

def plot_overlaps(m0, m1, title, filename):
    a = sim(m0, m1)
    ax = pd.DataFrame(a)
    ax.index = np.argmax(a, axis=1)
    ax = ax.sort()

    plt.figure()
    sns.heatmap(ax, cbar=True, xticklabels=ax.columns, yticklabels=ax.index, annot=False)
    plt.title(title)
    plt.savefig(filename)

# working.

words = 20
for topics in [20, 40, 75]:

    ix_a = create_sample(features, 0.8)
    ix_b = create_sample(features, 0.8)

    ma = get_word_assignments(vectorizer, run_on_sample(features, ix_a, n_topics=topics, n_iter=200), words)

    mb = get_word_assignments(vectorizer, run_on_sample(features, ix_b, n_topics=topics, n_iter=200), words)

    title = 'Overlap between %s most probable words with %s topics' % (words, topics)
    plot_overlaps(ma, mb, title, 'overlaps-%s-%s.png' % (topics, words))


## using 100 words, because, realistically, you would only use maybe 20 words to distinguish topics.
## TODO: try with 20 words too.


m20_80_a = get_word_assignments(vectorizer, run_on_sample(features, ix_a, n_topics=20, n_iter=100), 200)
m20_80_b = get_word_assignments(vectorizer, run_on_sample(features, ix_b, n_topics=20, n_iter=100), 200)



## I AM HERE: clean up this code, run for these different ones.

plot_overlaps(m75_80_a, m75_80_b, 'overlaps-75.png')
plot_overlaps(m40_80_a, m40_80_b, 'overlaps-40.png')

## NEXT: do random assignments of words: for each topic, draw 200 words from the vocab randomly, check overlap.


##NEXT: explore and make plots.

a = sim(m40_80_a, m40_80_b)

ax = pd.DataFrame(a)
ax.index = np.argmax(a, axis=1)
ax = ax.sort(axis='index')
sns.heatmap(ax, cbar=False, xticklabels=ax.columns, yticklabels=ax.index, annot=False)
plt.savefig('overlaps-sort-1.png')


a = sim(m40_80_a, m40_80_b)
ax = pd.DataFrame(a)
ax.index = np.argmax(a, axis=0)
ax = ax.sort(axis='index')
sns.heatmap(ax, cbar=False, xticklabels=ax.columns, yticklabels=ax.index, annot=False)
plt.savefig('overlaps-sort-0.png')





I AM HERE: I think this with the data frame might work.


sns.heatmap(a[np.argmax(a, axis=0),:], cbar=False)
plt.savefig('overlaps-words-40-80-sorted-nocbar.png')

sns.heatmap(np.argsort()a, cbar=False)
plt.savefig('overlaps-words-40-80-nocbar.png')


evaluate these heatmaps in some way.

## sort one dim by the argmax of the other.






## TODO: use this for words maybe?
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
