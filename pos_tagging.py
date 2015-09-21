"Improve feature extraction using pos tagging."

import collections
import itertools
import nltk
import operator
import pandas as pd
import re

import functions as f

df = f.load_data()

## tokenize
sents = df['ingredient_txt'].map(lambda x: map(nltk.word_tokenize, x.split('\n')))
## remove first and last elements, which are empty lists.
sents = map(lambda x: x[1:-1], sents)

tagged = [nltk.pos_tag_sents(x) for x in sents[0:1000]]

## trying named entity recognition.
nltk.ne_chunk(tagged[0])


## trying hand-coded identification of ingredients.

def seq(pos, x): return [t[pos] for t in x]

## split tokens and tags into separate lists.
tok_seq = map(lambda x: map(lambda xx: seq(0, xx), x), tagged)
tag_seq = map(lambda x: map(lambda xx: seq(1, xx), x), tagged)

## create mapping between flattened list of ingredients and recipe ids.
idx = {}
j = 0
for i,x in enumerate(tok_seq):
    for y in x:
        idx[j] = i
        j += 1


## flatten nested lists
tok_seq = list(itertools.chain(*tok_seq))
tag_seq = list(itertools.chain(*tag_seq))

## join list of tokens and tags into strings for hashing.
tok_seq = map(lambda x: re.sub('[,.]', '', ' '.join(x), ), tok_seq)
tag_seq = map(lambda x: re.sub('[,.]', '', ' '.join(x), ), tag_seq)


## create data frames useful for analysis of sequence frequency.
tt = pd.DataFrame({'tags': tag_seq, 'tokens': tok_seq})
freqs = tt.groupby('tags').agg(len).reset_index()
freqs.columns = ['tags','freq']
tt = pd.merge(tt, freqs)

## how much can I cover with coded sequences?
freqs.sort('freq').tail(int(0.3*freqs.shape[0])).sum()[1]/float(freqs['freq'].sum())
## coding 30%

## trying greedy match on subsequences.

## hammer approach: just extract nouns:
## I don't think it will work, because its more about pos and sequence.: 'teaspoons' is a noun, etc.

## TODO: trying a 3/4/5 topic model to id amounts 

## collocation approach:
f = nltk.BigramCollocationFinder.from_words(tag_seq[0:100])
bm = nltk.collocations.BigramAssocMeasures()
f.nbest(bm.pmi, 10)

## Trying subsequences
encodings = {
    ('NN', 'NN') : ['NA','NA']
}

## find allo 

## maybe compare bigram freq to other freqs: most common is actual word.

## only look at nouns
keep = set(['NN'])
ns = [[token for token,tag in ts if tag in keep] for ts in tagged[0]]



## NYT uses:
## NAME, UNIT, QUANTITY, COMMENT, OTHER
## 1 (QT) garlic clove (NA) , minced (optional) (OT)

## manually encode sequences with sequences.
encodings = {
      'CD NN NN NN': ['QT', 'UN', 'NA', 'NA']
    , 'LS NN NN NN': ['QT', 'UN', 'NA', 'NA']
    , 'CD NN NN': ['QT', 'UN', 'NA']
    , 'LS NN NN': ['QT', 'UN', 'NA']
    , 'NNP VBD': ['NA','NA']
    , 'CD NN JJ NN': ['QT', 'UN', 'NA', 'NA']
}

## associate tokens with ingredient types.
## exact matching.
matched = [zip(tokens.split(' '), encodings.get(tags, [])) for tags, tokens in zip(tag_seq, tok_seq)]

## TODO: do some greedy matching on the sequences with no frequent sequence matching.

## TODO: match ingredients back to recipes.

ingredients = map(lambda x: ' '.join([t for t,tag in x if tag=='NA']), matched)

im = [(idx[i], x) for i,x in enumerate(ingredients)]
imm = collections.defaultdict(lambda: ' ') 
for k,v in im:
    imm[k] += v

imd = pd.DataFrame([imm])
imd = pd.melt(imd)
imd.columns = ['i','ingredient_clean']

dff = imd.join(df)

    # 1: run and code for much larger sample.
    # 2: implement greedy matching.





## find most common tag sequences:

## make mapping between tag sequences and actual texts for checking.

ss = map(lambda x: ' '.join(x), ss)

seq_counts = collections.defaultdict(int)
for s in ss:
    seq_counts[s] += 1

del(seq_counts[''])

sorted_ss = sorted(seq_counts.items(), key=operator.itemgetter(1))

x = pd.DataFrame(sorted_ss, columns=['seq','freq'])

## show examples 


## TODO: look at the distribution of tags, see if there are some very common ones that we can code manually.






def tag_filter(iterable, tags):
    return [(token, tag) for (token, tag) in iterable if tag in tags]


## NYT uses:
## NAME, UNIT, QUANTITY, COMMENT, OTHER
## 1 (QT) garlic clove (NA) , minced (optional) (OT)

ingredients = set(['JJ','NN'])


## TODO: figure out what the ingredients are
