import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

epochs = 50
batch_size = 100

num_hidden_1 = 300
num_hidden_2 = 300

weights = {
	'hidden_1' : tf.Variable(tf.random_normal([784,num_hidden_1])),
	'hidden_2' : tf.Variable(tf.random_normal([num_hidden_1,num_hidden_2])),
	'output' : tf.Variable(tf.random_normal([num_hidden_2,10]))
}
biases = {
	'hidden_1' : tf.Variable(tf.random_normal([num_hidden_1])),
	'hidden_2' : tf.Variable(tf.random_normal([num_hidden_2])),
	'output' : tf.Variable(tf.random_normal([10]))
}

input_feature = tf.placeholder(tf.float32,[None,784])
input_label = tf.placeholder(tf.float32,[None,10])

def mulilayerperception(input,weights,biases):
	hidden_layer_1 = tf.nn.relu(tf.add(tf.matmul(input,weights['hidden_1']),biases['hidden_1']))
	hidden_layer_2 = tf.nn.relu(tf.add(tf.matmul(hidden_layer_1,weights['hidden_2']),biases['hidden_2']))
	output_layer = tf.add(tf.matmul(hidden_layer_2,weights['output']),biases['output'])
	return output_layer

predict_label = mulilayerperception(input_feature,weights,biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict_label,input_label))
optimizer = tf.train.AdamOptimizer(0.02).minimize(cost)

correct = tf.cast(tf.equal(tf.argmax(predict_label,1),tf.argmax(input_label,1)),tf.float32)
correct_rate = tf.reduce_mean(correct)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

batch_total = mnist.train.num_examples//batch_size
for epoch in range(epochs):
	cost_sum = 0.0
	for batch in range(batch_total):
		batch_feature , batch_label = mnist.train.next_batch(batch_size)
		_ , cost_receive = sess.run([optimizer, cost], feed_dict={input_feature:batch_feature, input_label:batch_label})
		cost_sum += cost_receive
	cost_avg = cost_sum/batch_total
	print("Epoch ", epoch , " Cost : ", cost_avg)

correct_result = sess.run(correct_rate,feed_dict={input_feature:mnist.test.images, input_label:mnist.test.labels})
print("Test Accuracy : ", correct_result)

sess.close()