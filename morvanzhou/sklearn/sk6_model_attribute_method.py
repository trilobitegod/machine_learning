# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 10:51:30 2019

@author: Snake
"""



from __future__ import print_function
from sklearn import datasets
from sklearn.linear_model import LinearRegression

loaded_data = datasets.load_boston()
X = loaded_data.data
y = loaded_data.target

model = LinearRegression()
model.fit(X, y)

print('coef   ', model.coef_)
print('inter  ', model.intercept_)
print('R^2    ', model.score(X, y))    # R^2
print(model.get_params())
print(len(X[1,:]))
