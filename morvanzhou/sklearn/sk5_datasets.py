# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 10:12:41 2019

@author: Snake
"""


from __future__ import print_function
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y = loaded_data.target

model = LinearRegression()
model.fit(data_X, data_y)


print(model.predict(data_X[:4,:]))
print(data_y[:4])


X, y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=10)
model2 = LinearRegression()
model2.fit(X, y)
y_predict = model2.predict(X)

plt.scatter(X, y)
plt.plot(X, y_predict, color='r')
plt.show()