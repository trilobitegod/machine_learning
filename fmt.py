import struct
from FullConnectedLayer import *
from datetime import datetime

def transpose(args):
	return list(map(lambda arg: arg.reshape(np.size(arg),1),args))

def ImageLoader(path):
	data = open(path, 'rb').read()
	#fmt of struct unpack, > means big endian, i means integer, well, iiii mean 4 integers
	fmt = '>iiii'
	offset = 0
	magic_number, img_number, height, width = struct.unpack_from(fmt, data, offset)
	print('magic number is {}, image number is {}, height is {} and width is {}'.format(magic_number, img_number, height, width))
	#slide over the 2 numbers above
	offset += struct.calcsize(fmt)
	#28x28
	image_size = height * width
	#B means unsigned char
	fmt = '>{}B'.format(image_size)
	
	images = np.empty((img_number, height, width))
	for i in range(img_number):
		images[i] = np.array(struct.unpack_from(fmt, data, offset)).reshape((height, width))
		offset += struct.calcsize(fmt)
	return images

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
	
	

train_data_set = transpose(ImageLoader('train-images.idx3-ubyte'))
train_labels = transpose(LabelLoader('train-labels.idx1-ubyte'))
test_data_set = transpose(ImageLoader('t10k-images.idx3-ubyte'))
test_labels = transpose(LabelLoader('t10k-labels.idx1-ubyte'))
'''
import matplotlib.pyplot as plt
plt.imshow(images[0])
plt.show()
'''

print(np.shape(train_data_set))
print(np.shape(train_labels))
print(np.shape(test_data_set))
print(np.shape(test_labels))
print(train_labels[0])