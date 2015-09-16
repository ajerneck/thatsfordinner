import bs4
import collections
import itertools
import urllib
import re
import requests
import pandas as pd
import sys
import psycopg2
from sqlalchemy import create_engine
from time import time

def make_soup(url):
    "Make a beautiful soup from an url."
    try:
        page = requests.get(url)
        soup = bs4.BeautifulSoup(page.text, 'html.parser')
        return soup
    except requests.ConnectionError as e:
        print str(e)
        return bs4.BeautifulSoup("", 'html.parser')


# main dishes last page: http://allrecipes.com/recipes/80/main-dish/?page=716#716

BASE_URL = "http://allrecipes.com/recipes/80/main-dish/?page=%s"

def parse_recipe_urls(url):
    "Scrape search results into list of recipe metadata."
    soup = make_soup(url)
    print(url)

    ## extract all data except the url.
    recipes = [x.attrs for x in soup.find_all(attrs={'data-type':"'Recipe'"})]

    ## get the url.
    rs = soup.select('article.grid-col--fixed-tiles')
    raw_urls = [x.find_all('a') for x in rs]
    urls = [x[0].attrs['href'] for x in raw_urls if x != []]

    ## combine metadata and url.
    for i,_ in enumerate(recipes):
        recipes[i][u'url'] = 'http://allrecipes.com' + urls[i]

    return recipes

def parse_recipe(recipe):
    "Scrape data for one recipe."
    print(recipe['url'])
    soup = make_soup(recipe['url'])
    recipe.update(dict.fromkeys(['ingredients','prep_time','cook_time','total_time','calories','ratings','reviews_nr']))
    try:
        recipe['ingredients'] = [x.text for x in soup.find_all('span', class_='recipe-ingred_txt added')]
        recipe['prep_time'] = soup.find(itemprop='prepTime').attrs['datetime']
        recipe['cook_time'] = soup.find(itemprop='cookTime').attrs['datetime']
        recipe['total_time'] = soup.find(itemprop='totalTime').attrs['datetime']
        recipe['calories'] = soup.find(class_='calorie-count').text

        ## nutrients
        ## TODO: parse nutrients.
        ## all_nutrients = soup.find_all(class_='nutrientLine')

        ## ratings
        raw_ratings = soup.find_all(title=re.compile('[0-9]+ cooks'))
        ratings = map(lambda x: int(x.split()[0]), [x.attrs['title'] for x in raw_ratings])
        recipe['ratings'] = dict(zip(range(5, 0, -1), ratings))
        recipe['reviews_nr'] = int(soup.find(class_='recipe-reviews__header--count').text)
    except:
        print 'Error: ', sys.exc_info()[0]


    return recipe

def make_dataframe(results):
    "Make dataframe from list of recipes."

    ## filter out results which did not parse.
    results = [x for x in results if x['ratings'] is not None]
    ## create separate dataframes for different attributes.

    ratings = pd.DataFrame([x['ratings'] for x in results])
    ratings.columns = ['rating_level_%s' % i for i in range(1,6)]
    ingredients = pd.DataFrame({'ingredients':'\n'.join(x['ingredients'])} for x in results)

    keys =  ['total_time', 'cook_time', u'url', 'calories', u'data-name', u'data-id', u'data-imageurl', 'prep_time', 'reviews_nr']
    metadata = pd.DataFrame([{key: x[key] for key in keys} for x in results])

    ## create one big data frame.
    return metadata.join(ingredients).join(ratings)

def save_dataframe(df):
    "Save dataframe to postgres and hd5."

    ## store as csv as well.
    df.to_hdf('allrecipes.h5', 'table')

    engine = create_engine("postgresql+psycopg2://explore:Ln2bOYAVCG6utNUSaSZaIVMH@localhost/explore")

    con = engine.connect()
    con.execute('drop table allrecipes')
    df.to_sql('allrecipes', con=engine)


## TODO: use full lists, not ranges.

main_dish_pages = [BASE_URL %  n  for n in range(1, 717)]
t_search_results0 = time()
main_dishes_search_results = map(parse_recipe_urls, main_dish_pages)
## flatten the nested list of search results.
main_dishes_search_results = list(itertools.chain.from_iterable(main_dishes_search_results))
print('Parsed %s search pages in %0.3f' % (len(main_dishes_search_results), time() - t_search_results0))

t_recipes0 = time()
main_dishes_recipes = map(parse_recipe, main_dishes_search_results)
print('Parsed %s recipes in %0.3f' % (len(main_dishes_recipes), time() - t_recipes0))

df = make_dataframe(main_dishes_recipes)
save_dataframe(df)

## check
e = make_engine()
xx = pd.read_sql_table('allrecipes', e)


# TODO: add more categories.


