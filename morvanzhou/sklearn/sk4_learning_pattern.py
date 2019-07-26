# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 09:58:27 2019

@author: Snake
"""

from __future__ import print_function
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)


knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print(knn.predict(X_test))
print(y_test)
print(knn.predict(X_test) == y_test)