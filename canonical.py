"Use a canonical list of ingredients or foods to match."

import nltk
import pandas as pd

import functions as f

import common

## exploiting that ingredients are mentioned in instructions as well.

con = common.make_engine()
dfe = pd.read_sql_table('recipes_recipe', con)
x = dfe.head()

## intersection of ingredients and instructions.
set(x['ingredient_txt'].str.split()[0]).intersection(set(x['instruction_txt'].str.split()[0]))


## olive oil
## parsley
## lemon peel
## garlick
## lemon juice
## kosher salt
## black pepper
## spinache and artichoke ravioli
## baby artichokes


## cannonical list attempt.

df = f.load_data()



## using canonical ingredient list.

cf = pd.read_csv('/home/alexander/start/archives/2015/2015-start/code/data-science/incubator-challenge/q3/fooddb/compounds_foods.csv', escapechar="\\")

vocab = cf['orig_food_common_name'].str.lower().unique()

## edit distances.
sents = df['ingredient_txt'].map(lambda x: map(nltk.word_tokenize, x.split('\n')))
sents = map(lambda x: x[1:-1], sents)

sents[0:10]

## simple approach: for the most probable words for each topic, if a unigram appears in a bigram, filter it out.

