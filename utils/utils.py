#####################################################
#
#
#   github - http://github.com/phraniiac
#   project - tensorflow boilerplate
#
#
#####################################################


import tensorflow as tf
import numpy as np


def minibatches(self, data, minibatch_size, shuffle=True):
	"""
	This function yields an iterator over the dataset of size - minibatch_size.
	
	Data can be nested lists inside of the actual list. 
	In case of nested list, all of the upper list wil contribute to
	the the minibatch by providing those indices.
	"""
	check_nested = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
	data_size = len(data[0]) if check_nested else len(data)
	indices = np.arange(data_size)
	if shuffle:
		np.random.shuffle(indices)
	for index in range(0, data_size, minibatch_size):
		m_indices = indices[index: index + minibatch_size]
		yield [minibatch(d, minibatch_indices) for d in data] if list_data \
			else minibatch(data, minibatch_indices)

def minibatch(data, minibatch_idx):
	# taken from stanford course cs224N assignment 2.
	return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]


def get_weights(self, name, shape, initializer='xavier', constant=0.0):
		if initializer == 'xavier':
			return tf.get_variable(name=name, shape=shape, \
					initializer=tf.contrib.layers.xavier_initializer())
		if initializer == 'constant':
			return tf.get_variable(name=name, shape=shape, \
					initializer=tf.constant_initializer(constant))

# Generic class to get weights and conv layers.

class layers_and_weights():
	def __init__(self):
		pass

	def FC_layer(self, x, W, b, op_type='relu'):
		return tf.nn.relu(tf.add(tf.matmul(x, W), b))

	def conv1d_layer(self, x, W, b, strides, padding='SAME'):
		conv_out = tf.nn.conv1d(x, W, strides, padding)
		return tf.add(conv_out, b)

	def get_weights(self, name, shape, initializer='xavier', constant=0.0):
		if initializer == 'xavier':
			return tf.get_variable(name=name, shape=shape, \
					initializer=tf.contrib.layers.xavier_initializer())
		if initializer == 'constant':
			return tf.get_variable(name=name, shape=shape, \
					initializer=tf.constant_initializer(constant))


# Class to get data for training.
class DataClass():
	def __init__(self):
		pass
	
	def get_batch_sized_data(batch_size, message_length):
		message_batch = np.random.uniform(-1.0, 1.0, [batch_size, message_length])
		return message_batch