#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import numpy as np
from functools import reduce
from activators import SigmoidActivator,IdentityActivator


class FullConnectedLayer(object):
	def __init__(self,input_size,output_size,activator):
		self.input_size = input_size
		self.output_size = output_size
		self.activator = activator
		self.W = np.random.uniform(-0.1,0.1,(output_size,input_size))
		self.b = np.zeros((output_size,1))
		self.output = np.zeros((output_size,1))
		
	def forward(self,input_array):
		self.input = input_array
		self.output = self.activator.forward(np.dot(self.W,input_array)+self.b)
		
	def backward(self,delta_array):
		self.delta = self.activator.backward(self.input)*np.dot(self.W.T,delta_array)
		self.W_grad = np.dot(delta_array,self.input.T)
		self.b_grad = delta_array
		
	def update(self,learning_rate):
		self.W += learning_rate*self.W_grad
		self.b += learning_rate*self.b_grad
		
	def dump(self):
		print('W: %s\nb:%s' % (self.W,self.b))
	
	def get_par(self):
		self.par = []
		self.par.append(self.W)
		self.par.append(self.b)
		return self.par
	
	
class Network(object):
	def __init__(self,layers):
		self.layers = []
		for i in range(len(layers)-1):
			self.layers.append(FullConnectedLayer(layers[i],layers[i+1],SigmoidActivator()))
			
	def predict(self,sample):
		output = sample
		for layer in self.layers:
			layer.forward(output)
			output = layer.output
		return output
			
	def train(self,labels,data_set,rate,epoch):
		for i in range(epoch):
			for d in range(len(data_set)):
				self.train_one_sample(labels[d],data_set[d],rate)
	
	def train_one_sample(self,label,sample,rate):
		self.predict(sample)
		self.calc_gradient(label)
		self.update_weight(rate)
		
	def calc_gradient(self,label):
		delta = self.layers[-1].activator.backward(self.layers[-1].output)*(label-self.layers[-1].output)
		for layer in self.layers[::-1]:
			layer.backward(delta)
			delta = layer.delta
		return delta
		
	def update_weight(self,rate):
		for layers in self.layers:
			layers.update(rate)
			
	def dump(self):
		for layer in self.layers:
			layer.dump()
			
	def loss(self,output,label):
		return 0.5*((label-output)*(label-output)).sum()
		
	def gradient_check(self,sample_feature,sample_label):
		self.predict(sample_feature)
		self.calc_gradient(sample_label)
		
		epsilon = 10e-4
		for fc in self.layers:
			for i in range(fc.W.shape[0]):
				for j in range(fc.W.shape[1]):
					fc.W[i,j] += epsilon
					output = self.predict(sample_feature)
					err1 = sele.loss(sample_label,output)
					fc.W[i,j] -= 2*epsilon
					output = self.predict(sample_feature)
					err2 = self.loss(sample_lbael,output)
					expect_grad = (err1-err2)/(2*epsilon)
					fc.W[i,j] += epsilon
					print('weights(%d,%d): expected - actural %.4e - %.4e' % (i,j,expect_grad,fc.W_grad[i,j]))
					
	def get_parameter(self):
		parameter = []
		for layer in self.layers:
			parameter.append(layer.get_par())
		return parameter
	
	
#from bp import train_data_set

def transpose(args):
	return list(map(lambda arg: arg.reshape(np.size(arg),1),args))
	
class Normalizer(object):
	def __init__(self):
		self.mask = [0x1,0x2,0x4,0x8,0x10,0x20,0x40,0x80]
	
	def norm(self,number):
		data = list(map(lambda m: 0.9 if number&m else 0.1,self.mask))
		return np.array(data).reshape(8,1)
		
	def denorm(self,vec):
		binary = list(map(lambda i: 1 if i>0.5 else 0,vec[:,0]))
		for i in range(len(self.mask)):
			binary[i] = binary[i]*self.mask[i]
		return reduce(lambda x,y: x+y,binary)
		
def train_data_set():
	normalizer = Normalizer()
	data_set = []
	labels = []
	for i in range(0,256):
		n = normalizer.norm(i)
		data_set.append(n)
		labels.append(n)
	return labels,data_set
	
def correct_ratio(network):
	normalizer = Normalizer()
	correct = 0.0
	for i in range(256):
		if normalizer.denorm(network.predict(normalizer.norm(i))) == i:
			correct += 1.0
	print('correct_ratio: %.2f%%' % (correct/256*100))
	
def test():
	labels,data_set = train_data_set()
	net = Network([8,3,8])
	rate = 0.5
	mini_batch = 20
	epoch = 10
	for i in range(epoch):
		net.train(labels,data_set,rate,mini_batch)
		print('after epoch %d loss: %f' % (i+1, net.loss(labels[-1],net.predict(data_set[-1]))))
		rate /= 2
	correct_ratio(net)
	
def gradient_check():
	labels,data_set = transpose(train_data_set())
	net = Network([8,3,8])
	net.gradient_check(data_set[0],labels[0])
	return net
			
