# -*- coding: utf-8 -*-
"""
Created on Wed May  1 15:07:45 2019

@author: Snake
"""

from urllib.request import urlopen
import re

html = urlopen(
        "https://morvanzhou.github.io/static/scraping/basic-structure.html"
        ).read().decode('utf-8')
print(html)

res = re.findall(r"<title>(.+?)</title>",html)
print("\nPage title is:",res[0])

res = re.findall(r"<p>(.*?)</p>",html,flags=re.DOTALL)  # re.DOTALL if multi line
print("\nPage paragraph is:", res[0])

res = re.findall(r'href="(.*?)"',html)
print("\nAll links:", res)