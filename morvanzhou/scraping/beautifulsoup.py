# -*- coding: utf-8 -*-
"""
Created on Wed May  1 15:21:37 2019

@author: Snake
"""

from bs4 import BeautifulSoup
from urllib.request import urlopen


html = urlopen("https://morvanzhou.github.io/static/scraping/basic-structure.html").read().decode('utf-8')
print(html)

soup = BeautifulSoup(html,features='html5lib')
print(soup.h1)
print('\n', soup.p)

all_href = soup.find_all('a')
all_href = [l['href'] for l in all_href]
print('\n', all_href)