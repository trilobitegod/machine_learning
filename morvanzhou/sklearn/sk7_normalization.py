# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


from __future__ import print_function
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_classification
from sklearn.svm import SVC
import matplotlib.pyplot as plt


X, y = make_classification(n_samples=300, n_features=2, n_redundant=0, n_informative=2,
                          random_state=22, n_clusters_per_class=1, scale=100)
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y)

X = preprocessing.scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
clf = SVC()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))


plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()