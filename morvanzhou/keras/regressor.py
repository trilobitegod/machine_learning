#!/usr/bin/env/python3
# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(1337)	# for reproducibility
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt	# 可视化模块

# create some data
x = np.linspace(-1,1,200)
np.random.shuffle(x)	# randomize data
y = 0.5*x+2+np.random.normal(0,0.05,(200,))

#plot data
plt.scatter(x,y)
plt.show()

x_train,y_train = x[:160],y[:160]
x_test,y_test = x[160:],y[160:]

model = Sequential()
model.add(Dense(output_dim = 1,input_dim = 1))

# choose loss function and optimizing method
model.compile(loss = 'mse',optimizer = 'sgd')

# training
print('Training ---------')
for step in range(301):
	cost = model.train_on_batch(x_train,y_train)
	if step % 100 == 0:
		print('train cost',cost)
		
print('\nTesting--------')
cost = model.evaluate(x_test,y_test,batch_size = 40)
print('test cost',cost)
W,b = model.layers[0].get_weights()
print('Weights=',W,'\nbiases=',b)

y_pred = model.predict(x_test)
plt.scatter(x_test,y_test)
plt.plot(x_test,y_pred)
plt.show()