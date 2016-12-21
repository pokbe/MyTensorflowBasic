import numpy as np
import tensorflow as tf

#https://www.tensorflow.org/versions/r0.12/tutorials/mnist/beginners/index.html#mnist-for-ml-beginners
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_feat, train_lab = mnist.train.next_batch(1000)
test_feat, test_lab = mnist.test.next_batch(200)

#print(type(train_feat)) #<class 'numpy.ndarray'>
print(train_feat.shape) #(1000, 784)
print(train_lab.shape) #(1000, 10)
print(test_feat.shape) #(400, 784)
print(test_lab.shape) #(400, 10)

input_train_feat = tf.placeholder(tf.float32, [None,784])
input_train_lab = tf.placeholder(tf.float32, [None,10])
input_test_feat = tf.placeholder(tf.float32, [784])

def model(train,label,test):
	dist = tf.reduce_sum(tf.abs(tf.add(train, tf.neg(test))), reduction_indices = 1)
	min_index = tf.argmin(dist,0)
	min_index = tf.cast(min_index, tf.int32)
	pred = label[min_index]
	return pred

predict = model(input_train_feat,input_train_lab,input_test_feat)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
countright = 0
for i in range(len(test_feat)):
	pre = sess.run(predict,feed_dict={input_train_feat:train_feat, input_train_lab:train_lab, input_test_feat:test_feat[i]})
	pre_index = tf.argmax(pre,0)
	true_index = tf.argmax(test_lab[i],0)
	#print(pre)
	print("Pre: ", sess.run(pre_index), " , True: ", sess.run(true_index))
	if(sess.run(pre_index) == sess.run(true_index)):
		countright += 1
print("Accuracy : ", countright/len(test_feat))

sess.close()

print("Done !!")