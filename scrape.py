import bs4
import collections
import itertools
import urllib
import requests

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
main_dish_pages = [BASE_URL %  n  for n in range(1, 717)]

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

main_dishes = map(parse_recipe_urls, main_dish_pages[0:10])



# TODO: scrape recipe page.

# TODO: add more categories.


