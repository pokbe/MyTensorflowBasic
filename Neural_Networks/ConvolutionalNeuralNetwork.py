import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

epochs = 100
batch_size = 100
kernel_size = 5
stride_size = 1
pool_size = 2

input_feature = tf.placeholder(tf.float32,[None,784])
input_label = tf.placeholder(tf.float32, [None,10])
keep_prob = tf.placeholder(tf.float32)

def conv_layer(input, weights, biases):
	conv_result = tf.nn.bias_add(tf.nn.conv2d(input = input, filter = weights, strides=[1,stride_size,stride_size,1], padding='SAME'),biases)
	relu_result = tf.nn.relu(conv_result)
	output = tf.nn.max_pool(value=relu_result, ksize=[1,pool_size,pool_size,1], strides=[1,pool_size,pool_size,1], padding='SAME')
	return output

def full_layer(input, weights, biases):
	linear_result = tf.add(tf.matmul(input, weights),biases)
	output = tf.nn.relu(linear_result)
	output_d = tf.nn.dropout(output,keep_prob=keep_prob)
	return output_d
def out_layer(input,weights, biases):
	linear_result = tf.add(tf.matmul(input, weights),biases)
	return linear_result

weights = {
	'conv_1' : tf.Variable(tf.random_normal([kernel_size,kernel_size,1,16])),
	'conv_2' : tf.Variable(tf.random_normal([kernel_size,kernel_size,16,64])),
	'full_1' : tf.Variable(tf.random_normal([7*7*64,1000])),
	'out' : tf.Variable(tf.random_normal([1000,10])),
}
biases = {
	'conv_1' : tf.Variable(tf.random_normal([16])),
	'conv_2' : tf.Variable(tf.random_normal([64])),
	'full_1' : tf.Variable(tf.random_normal([1000])),
	'out' : tf.Variable(tf.random_normal([10]))
}

reshape_feature = tf.reshape(input_feature,[-1,28,28,1]) 
conv_1 = conv_layer(reshape_feature, weights['conv_1'], biases['conv_1'])
conv_2 = conv_layer(conv_1, weights['conv_2'], biases['conv_2'])
reshape_conv = tf.reshape(conv_2, [-1,7*7*64])
full_1 = full_layer(reshape_conv, weights['full_1'], biases['full_1'])
full_2 = out_layer(full_1,weights['out'],biases['out'])

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(full_2, input_label))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

correct = tf.cast(tf.equal(tf.argmax(full_2,1),tf.argmax(input_label,1)),tf.float32)
correct_rate = tf.reduce_mean(correct)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

batch_total = mnist.train.num_examples//batch_size
for epoch in range(epochs):
	total_cost = 0.0
	for batch in range(batch_total):
		batch_feature, batch_label = mnist.train.next_batch(batch_size)
		_ , batch_cost,cor_rate = sess.run([optimizer,cost,correct_rate],feed_dict={input_feature:batch_feature, input_label:batch_label,keep_prob:0.6})
		total_cost += batch_cost
	avg_cost = total_cost/batch_total
	print("Epoch ",epoch," : Cost- ",avg_cost, "Correct- ",cor_rate)
print("Training Done!")

test_rate = sess.run(correct_rate,feed_dict={input_feature:mnist.test.images, input_label:mnist.test.labels, keep_prob:1.0})
print("Test correct rate : ", test_rate)
sess.close()

print("Done")