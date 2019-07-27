#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 07:58:57 2019

@author: trilobite
"""


from __future__ import print_function
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(knn.score(X_test, y_pred))


# this is cross_val_score
from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
print(scores)


#
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
k_range = range(1,31)
k_score = []
k_loss = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, scoring='accuracy')
    loss = -cross_val_score(knn, X, y, scoring='neg_mean_squared_error')
    k_score.append(scores.mean())
    k_loss.append(loss.mean())
    
fig, left_axis = plt.subplots()
right_axis = left_axis.twinx()    
left_axis.plot(k_range, k_score, color='r')
right_axis.plot(k_range, k_loss, color='b')
plt.show()