# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
import numpy as np

X_train = np.random.rand(100).astype(np.float32)
y_train = X_train*0.1+0.3

Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
bias = tf.Variable(tf.zeros([1]))
y = Weights*X_train+bias
loss =  tf.reduce_mean(tf.square(y-y_train))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(301):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(Weights),sess.run(bias))