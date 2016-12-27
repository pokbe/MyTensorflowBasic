import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

epochs = 15
batch_size = 100

num_input = 784
num_hidden_1 = 300
num_hidden_2 = 150

input_feature = tf.placeholder(tf.float32,[None,num_input])

weights = {
	'encode_hidden_1' : tf.Variable(tf.random_normal([num_input,num_hidden_1])),
	'encode_hidden_2' : tf.Variable(tf.random_normal([num_hidden_1,num_hidden_2])),
	'decode_hidden_1' : tf.Variable(tf.random_normal([num_hidden_2,num_hidden_1])),
	'decode_hidden_2' : tf.Variable(tf.random_normal([num_hidden_1,num_input]))
}
biases = {
	'encode_hidden_1' : tf.Variable(tf.random_normal([num_hidden_1])),
	'encode_hidden_2' : tf.Variable(tf.random_normal([num_hidden_2])),
	'decode_hidden_1' : tf.Variable(tf.random_normal([num_hidden_1])),
	'decode_hidden_2' : tf.Variable(tf.random_normal([num_input]))
}

def encoder_model(input_raw, w, b):
	output_1 = tf.nn.sigmoid(tf.add(tf.matmul(input_raw,w['encode_hidden_1']), b['encode_hidden_1']))
	output_2 = tf.nn.sigmoid(tf.add(tf.matmul(output_1,w['encode_hidden_2']), b['encode_hidden_2']))
	return output_2

def decoder_model(output_raw, w, b):
	revert_1 = tf.nn.sigmoid(tf.add(tf.matmul(output_raw,w['decode_hidden_1']), b['decode_hidden_1']))
	revert_2 = tf.nn.sigmoid(tf.add(tf.matmul(revert_1,w['decode_hidden_2']), b['decode_hidden_2']))
	return revert_2

encode_feature = encoder_model(input_feature, weights, biases)
revert_feature = decoder_model(encode_feature, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(revert_feature, input_feature))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

