import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

epochs = 100
batch_size = 100

num_hidden_1 = 300
num_hidden_2 = 200

weights = {
	hidden_1 : tf.Variable(tf.random_normal(784,num_hidden_1))
	hidden_2 : tf.Variable(tf.random_normal(num_hidden_1,num_hidden_2))
	output : tf.Variable(tf.random_normal(num_hidden_2,10))
}
biases = {
	hidden_1 : tf.Variable(tf.random_normal(num_hidden_1))
	hidden_2 : tf.Variable(tf.random_normal(num_hidden_2))
	output : tf.Variable(tf.random_normal(10))
}

input_feature = tf.placeholder(tf.float32,[None,784])
input_label = tf.placeholder(tf.float32,[None,10])

def mulilayerperception(input,weights,biases):
	hidden_layer_1 = tf.nn.relu(tf.add(tf.matmul(input,weights[hidden_1]),biases[hidden_1]))
	hidden_layer_2 = tf.nn.relu(tf.add(tf.matmul(hidden_layer_1,weights[hidden_2]),biases[hidden_2]))
	output_layer = tf.add(tf.matmul(hidden_layer_2,weights[output]),biases[output])
	return output

predict_label = mulilayerperception(input_feature,weights,biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict_label,input_label))
optimizer = tf.train.AdamOptimizer(0.02).minimize(cost)

