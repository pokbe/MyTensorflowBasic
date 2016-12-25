import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

epochs = 10
batch_size = 100

num_input = 28
num_step = 28
num_hidden = 100
num_class = 10

