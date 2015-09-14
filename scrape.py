import bs4
import collections
import itertools
import urllib
import re
import requests
import pandas as pd

def make_soup(url):
    "Make a beautiful soup from an url."
#    page = requests.get(url)
#    return bs4.BeautifulSoup(page.text, 'html.parser')
    try:
        page = requests.get(url)
        soup = bs4.BeautifulSoup(page.text, 'html.parser')
        return soup
    except requests.ConnectionError as e:
        print str(e)
        return bs4.BeautifulSoup("", 'html.parser')


# main dishes last page: http://allrecipes.com/recipes/80/main-dish/?page=716#716

# TODO: generate search urls
BASE_URL = "http://allrecipes.com/recipes/80/main-dish/?page=%s"

# TODO: scrape search result pages.

def parse_recipe_urls(url):
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
    print(recipe['url'])
    soup = make_soup(recipe['url'])
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

    return recipe

## TODO: use full lists, not ranges.
main_dish_pages = [BASE_URL %  n  for n in range(1, 717)]
main_dishes_search_results = map(parse_recipe_urls, main_dish_pages[::100])
## flatten the nested list of search results.
main_dishes_search_results = list(itertools.chain.from_iterable(main_dishes_search_results))

main_dishes_recipes = map(parse_recipe, main_dishes_search_results[::10])

## build data frame.
## ingredients:
[' '.join(x['ingredients']) for x in main_dishes_recipes]

## create separate dataframes for different attributes.
ratings = pd.DataFrame([x['ratings'] for x in main_dishes_recipes])
ratings.columns = ['rating_level_%s' % i for i in range(1,6)]
ingredients = pd.DataFrame({'ingredients':'\n'.join(x['ingredients'])} for x in main_dishes_recipes)

keys =  ['total_time', 'cook_time', u'url', 'calories', u'data-name', u'data-id', u'data-imageurl', 'prep_time', 'reviews_nr']
metadata = pd.DataFrame([{key: x[key] for key in keys} for x in main_dishes_recipes])


df = metadata.join(ingredients).join(ratings)



# TODO: add more categories.


