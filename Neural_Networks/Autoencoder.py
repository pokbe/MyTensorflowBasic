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

