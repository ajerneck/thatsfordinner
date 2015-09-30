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

import evaluate_functions as ef

df = p.load_data()
df = p.clean_formatting(df)
df = p.remove_stopwords(df)

assert(type(df)) is pd.core.frame.DataFrame, "%r is not a DataFrame." % df
assert(df.shape) == (16526, 7), "Has the wrong shape."

vectorizer, features = p.extract_features(df, title=True)

## set overlap (jaccard index) between the 20 most probable words on two different 80% parts of the original dataset.

words = 20
for topics in [20, 40, 75]:

    ix_a = ef.create_sample(features, 0.8)
    ix_b = ef.create_sample(features, 0.8)

    ma = ef.get_word_assignments(vectorizer, ef.run_on_sample(features, ix_a, n_topics=topics, n_iter=200), words)

    mb = ef.get_word_assignments(vectorizer, ef.run_on_sample(features, ix_b, n_topics=topics, n_iter=200), words)

    print 'runnig'
    title = 'Overlap between %s most probable words with %s topics' % (words, topics)
    ef.plot_overlaps(ma, mb, title, 'overlaps-%s-%s.png' % (topics, words))

## using 20 words, because, realistically, you would only use maybe 20 words to distinguish topics.
