# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 00:58:41 2018

@author: Snake
"""



import numpy as np
np.random.seed(1337) # for reproducibility
from keras.utils import np_utils
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense,Activation,Convolution2D,MaxPooling2D,Flatten


(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1,1,28,28) # normalize
X_test = X_test.reshape(-1,1,28,28) # normalize



y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

#......................................................................................
print('X_train shape: ',X_train.shape)
print('X_test shape: ',np.shape(X_test ))

model = Sequential()

model.add(Convolution2D(
        batch_input_shape=(32,1,28,28),
        filters=32,
        kernel_size=5,
        strides=1,
        padding='same',
        data_format='channels_last',))
model.add(Activation('relu'))
model.add(MaxPooling2D(
        pool_size=2,
        strides=2,
        padding='same',
        data_format='channels_last'))
model.add(Convolution2D(64,5,strides=1,padding='same',data_format='channels_last'))
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2,'same',data_format='channels_last'))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('\nTraining ------------')
model.fit(X_train, y_train, batch_size=32, epochs=2)
