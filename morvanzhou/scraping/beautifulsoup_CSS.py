# -*- coding: utf-8 -*-
"""
Created on Wed May  1 15:41:08 2019

@author: Snake
"""

from bs4 import BeautifulSoup
from urllib.request import urlopen
import re


html = urlopen("https://morvanzhou.github.io/static/scraping/list.html").read().decode('utf-8')
print(html)

soup = BeautifulSoup(html, features="html5lib")
month = soup.find_all('li', {"class": "month"})
for m in month:
    print(m.get_text())
    
jan = soup.find('ul', {"class": 'jan'})
date = jan.find_all('li')
for d in date:
    print(d.get_text())
print('\n')    

# bs4+正则表达式
html = urlopen("https://morvanzhou.github.io/static/scraping/table.html").read().decode('utf-8')
print(html)
soup = BeautifulSoup(html, features='html5lib')
image = soup.find_all('img', {"src": re.compile('.*?\.jpg')})
print('\n')
for i_jpg in image:
    print(i_jpg['src'])
    
course_links = soup.find_all("a", {"href": re.compile('https://morvanzhou.*')})
print('\n')
for couse_link in course_links:
    print(couse_link['href'])