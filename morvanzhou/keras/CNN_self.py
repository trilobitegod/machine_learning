# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 00:35:22 2018

@author: Snake
"""

import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Activation,Convolution2D,MaxPooling2D,Dense,Flatten

(X_train,y_train),(X_test,y_test) = mnist.load_data()
X_train = X_train.reshape(-1,1,28,28)
X_test = X_test.reshape(-1,1,28,28)
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)

model = Sequential()

model.add((Convolution2D(
        batch_input_shape=(32,1,28,28),
        filters=32,
        kernel_size=5,
        strides=1,
        padding='same',
        data_format='channels_last')))
model.add(Activation('relu'))
model.add(MaxPooling2D(
        pool_size=2,
        strides=2,
        padding='same',
        data_format='channels_last'))
model.add((Convolution2D(64,5,strides=1,padding='same',data_format='channels_last')))
model.add(Activation('relu'))
model.add(MaxPooling2D(2,strides=2,padding='same',data_format='channels_last'))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('\nTraining------------')
model.fit(X_train,y_train,batch_size=32,epochs=2)

loss,accuracy = model.evaluate(X_test,y_test)
print('\nTesting------------')
print('test loss: %.3f' % loss)
print('test accuracy: %.3f' % accuracy)