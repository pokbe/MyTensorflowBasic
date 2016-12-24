import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

epochs = 10
batch_size = 100

num_input = 28
num_step = 28
num_class = 10
num_hidden = 100

input_feature = tf.placeholder(tf.float32,[None, num_step*num_input])
input_feature_reshape = tf.reshape(input_feature,[-1,num_step,num_input])
input_label = tf.placeholder(tf.float32,[None,num_class])

weights = {
	'out' : tf.Variable(tf.random_normal([num_hidden,num_class])) 
}
biases = {
	'out' : tf.Variable(tf.random_normal([num_class]))
}

def LSTM_model(input_raw,w,b):
	# Current data input shape: (batch_size, n_steps, n_input)
	# Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
	input_list = tf.unpack(input_raw, axis=1)
	lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, forget_bias=0.9)
	outputs, state = tf.nn.rnn(lstm_cell, inputs = input_list, dtype = tf.float32) #outputs is the shape of len(outputs)=num_step outputs[-1].get_shape()=(num_batch,num_hidden) 
	# Returns:A pair (outputs, state) where: - outputs is a length T list of outputs (one for each input), or a nested tuple of such elements. - state is the final state
	result = tf.add(tf.matmul(outputs[-1],w['out']),b['out'])
	return result

predict_label = LSTM_model(input_feature_reshape,weights,biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict_label,input_label))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

correct = tf.cast(tf.equal(tf.argmax(predict_label,1),tf.argmax(input_label,1)),tf.float32)
correct_rate = tf.reduce_mean(correct)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
batch_total = mnist.train.num_examples//batch_size
for epoch in range(epochs):
	cost_sum = 0.0
	for batch in range(batch_total):
		batch_feature , batch_label = mnist.train.next_batch(batch_size)
		_ , cost_receive,cor_rate = sess.run([optimizer, cost, correct_rate], feed_dict={input_feature:batch_feature, input_label:batch_label})
		cost_sum += cost_receive
	cost_avg = cost_sum/batch_total
	print("Epoch ", epoch , " Cost : ", cost_avg, "Correct rate: ",cor_rate)
print("Training Done!")

correct_result = sess.run(correct_rate,feed_dict={input_feature:mnist.test.images, input_label:mnist.test.labels})
print("Test Accuracy : ", correct_result)
sess.close()
print("Done!")
'''
Epoch  0  Cost :  0.458767133897 Correct rate:  0.97
Epoch  1  Cost :  0.133187524693 Correct rate:  0.99
Epoch  2  Cost :  0.0875643024238 Correct rate:  0.97
Epoch  3  Cost :  0.0699509679306 Correct rate:  0.99
Epoch  4  Cost :  0.0563390860491 Correct rate:  0.98
Epoch  5  Cost :  0.0469590079992 Correct rate:  0.98
Epoch  6  Cost :  0.0413167397517 Correct rate:  1.0
Epoch  7  Cost :  0.0347070854077 Correct rate:  0.99
Epoch  8  Cost :  0.0350513017605 Correct rate:  0.98
Epoch  9  Cost :  0.0262284786336 Correct rate:  0.98
Training Done!
Test Accuracy :  0.982
Done!
'''