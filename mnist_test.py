#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import struct
import random
import numpy as np
from functools import reduce
from activators import SigmoidActivator,IdentityActivator
import matplotlib.pyplot as plt

class FullConnectedLayer(object):
	def __init__(self,W,b,activator):
		self.activator = activator
		self.W = W
		self.b = b
		self.output = np.zeros((len(b),1))
		
	def forward(self,input_array):
		self.input = input_array
		self.output = self.activator.forward(np.dot(self.W,input_array)+self.b)
		
	
class Network(object):
	def __init__(self,Wbs):
		self.layers = []
		for i in range(len(Wbs)):
			self.layers.append(FullConnectedLayer(Wbs[i][0],Wbs[i][1],SigmoidActivator()))
			
	def predict(self,sample):
		output = sample
		for layer in self.layers:
			layer.forward(output)
			output = layer.output
		return output

def get_result(vec):
    max_value_index = 0
    max_value = 0
    for i in range(len(vec)):
        if vec[i] > max_value:
            max_value = vec[i]
            max_value_index = i
    return max_value_index
	
def evaluate(network, test_data_set, test_labels):
    error = 0
    total = len(test_data_set)
    for i in range(total):
        label = get_result(test_labels[i])
        predict = get_result(network.predict(test_data_set[i]))
        if label != predict:
            error += 1
    return float(error) / float(total)
	
def transpose(args):
	return list(map(lambda arg: np.reshape(arg,(np.size(arg),1)),args))

def ImageLoader(filename):
	binfile = open(filename,'rb')
	buffers = binfile.read()
	head = struct.unpack_from('>IIII',buffers,0)
	offset = struct.calcsize('>IIII')
	imgNum = head[1]
	width = head[2]
	height = head[3]
	bits = imgNum*width*height
	bitsString = '>' + str(bits) + 'B'
	imgs = struct.unpack_from(bitsString,buffers,offset)
	binfile.close()
	imgs = np.reshape(imgs,(imgNum,width,height))
	return imgs
	
def LabelLoader(path):
	binfile = open(path,'rb')
	data = binfile.read()
	head = struct.unpack_from('>II',data,0)
	labelNum = head[1]
	offset = struct.calcsize('>II')
	numString = '>' + str(labelNum) + "B"
	labels = struct.unpack_from(numString,data,offset)
	binfile.close()
	labels = np.reshape(labels,[labelNum])
	label_vec = []
	for label in labels:
		label_int = int(label)
		for d in range(10):
			if label_int == d:
				label_vec.append(0.9)
			else:
				label_vec.append(0.1)
	return np.reshape(label_vec,(len(labels),10))
	
	
def train_and_evaluate():
	Wbs = np.load('parameters.npy')
	test_data_set = transpose(ImageLoader('t10k-images.idx3-ubyte'))
	test_labels = transpose(LabelLoader('t10k-labels.idx1-ubyte'))
	network = Network(Wbs)
	error_ratio = evaluate(network, test_data_set, test_labels)
	print('error ratio is %f' % error_ratio)

	
Wbs = np.load('parameters.npy')
imgs = ImageLoader('t10k-images.idx3-ubyte')
test_data_set = transpose(imgs)
test_labels = transpose(LabelLoader('t10k-labels.idx1-ubyte'))
network = Network(Wbs)
test_data_predict = []
test_label = []
test_data_fail = []
for i in range(213):
	test_data_predicti = get_result(network.predict(test_data_set[i]))
	test_data_labeli = get_result(test_labels[i])
	test_data_predict.append(test_data_predicti)
	test_label.append(test_data_labeli)
	if test_data_predicti != test_data_labeli:
		test_data_fail.append(test_data_predicti)
		plt.figure(i)
		plt.imshow(imgs[i])

print(test_data_predict)
print(test_label)
print(test_data_fail)
plt.show()