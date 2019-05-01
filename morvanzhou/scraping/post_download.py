# -*- coding: utf-8 -*-
"""
Created on Wed May  1 17:31:13 2019

@author: Snake
"""

from bs4 import BeautifulSoup
from urllib.request import urlretrieve
import requests
import webbrowser

'''
data = {'firstname': 'morvan',  'lastname': 'zhou'}
r = requests.post('http://pythonscraping.com/pages/files/processing.php',data=data)
print(r.text)

file = {'uploadFile': open('./image.jpg', 'rb')}
r = requests.post('http://pythonscraping.com/pages/files/processing2.php', files=file)
print(r.text)

session = requests.Session()
payload = {'username':'Morvan','password':'password'}
r = session.post('http://pythonscraping.com/pages/cookies/welcome.php',data=payload)
print(r.cookies.get_dict())

# session保存cookies，不用再传递data
r = session.post('http://pythonscraping.com/pages/cookies/welcome.php')
print(r.text)


#下载文件
IMAGE_URL = "https://morvanzhou.github.io/static/img/description/learning_step_flowchart.png"
urlretrieve(IMAGE_URL,'./img/image1.png')
r = requests.get(IMAGE_URL)
with open('./img/image2.png','wb') as f:
    f.write(r.content)
'''

#国家地理下载图片    
URL = 'http://www.ngchina.com.cn/animals/'
html = requests.get(URL).text
soup = BeautifulSoup(html, features='html5lib')
imgs = soup.find_all("div",{"class":"photo_story_up"})
for img in imgs:
    photo = img.find_all("img",{"title":"photo story"})
    url = photo[0]['src']
    r = requests.get(url, stream=True)
    photo_name = url.split('/')[-1]
    with open('./img/%s' % photo_name, 'wb') as f:
        for chunk in r.iter_content(chunk_size=128):
            f.write(chunk)
        print('Saved %s' % photo_name)


#国家地理下载视频
URL = 'http://www.ngchina.com.cn/animals/photo/4822.html###'
html = requests.get(URL).text
soup = BeautifulSoup(html, features='html5lib')
videos = soup.find_all("video",{"autoplay":"autoplay"})
for video in videos:
    #photo = img.find_all("img",{"title":"photo story"})
    if len(videos) == 1:
        url = video['src']
    elif len(videos) >= 1:
        url = video[0]['src']
    r = requests.get(url, stream=True)
    photo_name = url.split('/')[-1]
    with open('./img/%s' % photo_name, 'wb') as f:
        for chunk in r.iter_content(chunk_size=128):
            f.write(chunk)
        print('Saved %s' % photo_name)