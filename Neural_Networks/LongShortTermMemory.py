import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

epochs = 50
batch_size = 100

num_input = 28
num_step = 29
num_class = 10
num_hidden = 100

input_feature = tf.placeholder(tf.float32,[None,num_step,num_input])
input_label = tf.placeholder(tf.float32,[None,num_class])

weights = {
	'out' : tf.Variable(tf.random_normal([num_hidden,num_class])) 
}
biases = {
	'out' : tf.Variable(tf,random_normal([num_class]))
}

def LSTM_model(input_raw):
	# Current data input shape: (batch_size, n_steps, n_input)
	# Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
	input_list = tf.unpack(input_raw, axis=1)
	lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, forget_bias=0.9)
	outputs, state = tf.nn.rnn(lstm_cell, inputs = input_list, dtype = tf.float32) #outputs is the shape of len(outputs)=num_step outputs[-1].get_shape()=(num_batch,num_hidden) 
	# Returns:A pair (outputs, state) where: - outputs is a length T list of outputs (one for each input), or a nested tuple of such elements. - state is the final state
	outputs 