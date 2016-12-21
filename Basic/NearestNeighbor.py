import numpy as np
import tensorflow as tf

#https://www.tensorflow.org/versions/r0.12/tutorials/mnist/beginners/index.html#mnist-for-ml-beginners
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_feat, train_lab = mnist.train.next_batch(1000)
test_feat, test_lab = mnist.test.next_batch(400)

#print(type(train_feat)) #<class 'numpy.ndarray'>
print(train_feat.shape) #(1000, 784)
print(train_lab.shape) #(1000, 10)
print(test_feat.shape) #(400, 784)
print(test_lab.shape) #(400, 10)

tf.placeholder()