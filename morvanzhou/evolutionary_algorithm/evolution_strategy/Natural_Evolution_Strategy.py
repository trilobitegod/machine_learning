# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 00:29:36 2019

@author: Snake
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.distributions import MultivariateNormalFullCovariance


DNA_SIZE = 2
N_POP = 20
N_GENERATION = 100
LR = 0.02


def get_fitness(pred): return -((pred[:, 00])**2 + pred[:, 1]**2)

mean = tf.Variable(tf.random_normal([2, ], 13., 1.), dtype=tf.float32)
cov = tf.Variable(5. * tf.eye(DNA_SIZE), dtype=tf.float32)
mvn = MultivariateNormalFullCovariance(loc=mean, covariance_matrix=cov)
make_kid = mvn.sample(N_POP)

tfkids_fit = tf.placeholder(tf.float32, [N_POP, ])
tfkids = tf.placeholder(tf.float32, [N_POP, DNA_SIZE])
loss = -tf.reduce_mean(mvn.log_prob(tfkids)*tfkids_fit)
train_op = tf.train.GradientDescentOptimizer(LR).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

n = 300
x = np.linspace(-20, 20, n)
X, Y = np.meshgrid(x, x)
Z = np.zeros_like(X)
for i in range(n):
    for j in range(n):
        Z[i, j] = get_fitness(np.array([[x[i], x[j]]]))
plt.contourf(X, Y, -Z, 100, cmap=plt.cm.rainbow)
plt.ylim(-20, 20)
plt.xlim(-20, 20)
plt.ion()

for g in range(N_GENERATION):
    kids = sess.run(make_kid)
    kids_fit = get_fitness(kids)
    sess.run(train_op, {tfkids_fit: kids_fit, tfkids: kids})
    
    if 'sca' in globals(): sca.remove()
    sca = plt.scatter(kids[:, 0], kids[:, 1], s=30, c='k')
    plt.pause(0.01)
    
print('Finished')
plt.ioff()
plt.show