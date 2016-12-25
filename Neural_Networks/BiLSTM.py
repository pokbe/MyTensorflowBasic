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

input_feature = tf.placeholder(tf.float32,[None,num_step*num_input])
input_feature_reshape = tf.reshape(input_feature,[-1,num_step,num_input])
input_label = tf.placeholder(tf.float32,[None,num_class])

weights = {
	'out' : tf.Variable(tf.random_normal([2*num_hidden,num_class]))
}
biases = {
	'out' : tf.Variable(tf.random_normal([num_class]))
}

def BiLSTM_model(input_raw, w, b):
	#tf.nn.rnn_cell.BasicLSTMCell.__init__(num_units, forget_bias=1.0, input_size=None, state_is_tuple=True, activation=tanh)
	#tf.nn.bidirectional_rnn(cell_fw, cell_bw, inputs, initial_state_fw=None, initial_state_bw=None, dtype=None, sequence_length=None, scope=None)
	#inputs: A length T list of inputs, each a tensor of shape [batch_size, input_size], or a nested tuple of such elements.
	input_list = tf.unpack(input_raw, axis=1)

	fw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, forget_bias=0.9)
	bw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, forget_bias=0.9)

	outputs, output_state_fw, output_state_bw = tf.nn.bidirectional_rnn(fw_cell, bw_cell, input_list, dtype = tf.float32)
	result = tf.add(tf.matmul(outputs, w['out']),b['out'])
	return result

predict_label = BiLSTM_model(input_feature_reshape, weights , biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict_label, input_label))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

correct = tf.cast(tf.equal(tf.argmax(predict_label, 1), tf.argmax(input_label, 1)) , tf.float32)
correct_rate = tf.reduce_mean(correct)

sess = tf.Session()
sess.run(initialize_all_variables())
batch_total = mnist.train.num_examples//batch_size
for epoch in range(epochs):
	cost_sum = 0.0
	for batch in range(batch_total):
		batch_feature,batch_label = mnist.train.next_batch(batch_size)
		_, cost_receive, cor_rate = sess.run([optimizer,cost,correct_rate],feed_dict={input_feature:batch_feature, input_label:batch_label})
		cost_sum += cost_receive
	cost_avg = cost_sum/batch_total
	print("Epoch ", epoch , " Cost : ", cost_avg, "Correct rate: ",cor_rate)
print("Training Done !!!")

correct_result = sess.run(correct_rate, feed_dict={input_feature:mnist.test.images, input_label:mnist.test.labels})
print("Test Accuracy : ", correct_result)
sess.close()

print("Done !!")