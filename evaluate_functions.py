
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

    print 'Overlap score: %.2f' % (np.mean(np.max(all_diffs, axis=1)))
    print 'Overlap score: %.2f' % (np.mean((all_diffs)) * dim)
    return all_diffs

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


