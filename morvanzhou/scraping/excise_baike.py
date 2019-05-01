# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:19:02 2019

@author: Snake
"""

from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
import random


base_url = "https://baike.baidu.com"
his = ["/item/%E7%BD%91%E7%BB%9C%E7%88%AC%E8%99%AB/5162711"]


for i in range(20):
    url = base_url + his[-1]
    html = urlopen(url).read().decode('utf-8')
    soup = BeautifulSoup(html, features='html5lib')
    print(i, soup.h1.get_text(), '   url:', his[-1])
    
    sub_urls = soup.find_all("a", {"target": "_blank", "href": re.compile("/item/(%.{2})+$")})
    if len(sub_urls) !=0:
        his.append(random.sample(sub_urls, 1)[0]['href'])
    else:
        his.pop()
        

