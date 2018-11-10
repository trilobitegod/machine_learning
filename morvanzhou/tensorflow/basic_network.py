# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 08:43:59 2018

@author: Snake
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs,input_size,output_size,n_layer,activation_function=None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([input_size,output_size]),name='W')
            tf.summary.histogram(layer_name+'/Weights',Weights)
        with tf.name_scope('bias'):
            bias = tf.Variable(tf.zeros([1,output_size])+0.1,name='b')
            tf.summary.histogram(layer_name+'/bias',bias)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs,Weights)+bias
    if activation_function == None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs
    
x_data = np.linspace(-1,1,100,dtype=np.float32)[:,np.newaxis]
noise = np.random.normal(-0.05,0.05,x_data.shape).astype(np.float32)
y_data = np.square(x_data)-0.5+noise

with tf.name_scope('inputs'):
    #droupout to solve overfitting
    keep_prob = tf.placeholder(tf.float32,name='keep_prob')
    xs = tf.placeholder(tf.float32,[None,1],name='x_input')
    ys = tf.placeholder(tf.float32,[None,1],name='y_input')

layer1 = add_layer(x_data,1,20,n_layer=1,activation_function=tf.nn.relu)
prediction = add_layer(layer1,20,1,n_layer=2,activation_function=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
    tf.summary.scalar('loss',loss)
with tf.name_scope('optimizer'):
    optm = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()


init = tf.global_variables_initializer()
sess.run(init)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/",sess.graph)
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)

for i in range(1000):
    sess.run(optm,feed_dict={xs:x_data,ys:y_data,keep_prob:0.5})
    if i % 50 == 0:
        result = sess.run(merged,feed_dict={xs:x_data,ys:y_data,keep_prob:0.5})
      #  writer.add_summary(result,i)
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_values = sess.run(prediction,feed_dict={xs:x_data})
        lines = ax.plot(x_data,prediction_values,'r-',lw=5)
        plt.pause(0.1)
        
        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        