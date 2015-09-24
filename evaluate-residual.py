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



