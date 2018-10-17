#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import struct
from fc import *
from datetime import datetime
import time

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
	imgs = np.reshape(imgs,(imgNum,width*height))
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

def train_and_evaluate():
	last_error_ratio = 1.0
	epoch = 0
	train_data_set = transpose(ImageLoader('train-images.idx3-ubyte'))
	train_labels = transpose(LabelLoader('train-labels.idx1-ubyte'))
	test_data_set = transpose(ImageLoader('t10k-images.idx3-ubyte'))
	test_labels = transpose(LabelLoader('t10k-labels.idx1-ubyte'))
	network = Network([784, 300, 10])
	while True:
		epoch += 1
		start = time.time()
		network.train(train_labels, train_data_set, 0.01, 1)
		print('%.4fs epoch %d finished, loss %f' % (float(time.time()-start), epoch, network.loss(train_labels[-1], network.predict(train_data_set[-1]))))
		if epoch % 2 == 0:
			np.save('parameters',network.get_parameter())
		if epoch % 2 == 0:
			error_ratio = evaluate(network, test_data_set, test_labels)
			print('%.4fs after epoch %d, error ratio is %f' % (float(time.time()-start), epoch, error_ratio))
			if error_ratio > last_error_ratio:
				np.save('parameters',network.get_parameter())
				break
			else:
				last_error_ratio = error_ratio

if __name__ == '__main__':
    train_and_evaluate()

