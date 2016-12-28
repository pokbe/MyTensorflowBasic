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

log_path = 'lstm_log/1/'

input_feature = tf.placeholder(tf.float32,[None, num_step*num_input], name='Feature')
input_feature_reshape = tf.reshape(input_feature,[-1,num_step,num_input], name='Reshape_Feature')
input_label = tf.placeholder(tf.float32,[None,num_class], name='Label')

weights = {
	'out' : tf.Variable(tf.random_normal([num_hidden,num_class]), name='Weights_out') 
}
biases = {
	'out' : tf.Variable(tf.random_normal([num_class]), name='Biases_out')
}

def LSTM_model(input_raw,w,b):
	with tf.name_scope('LSTN_Model'):
		# Current data input shape: (batch_size, n_steps, n_input)
		# Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
		input_list = tf.unpack(input_raw, axis=1)
		lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, forget_bias=0.9)
		outputs, state = tf.nn.rnn(lstm_cell, inputs = input_list, dtype = tf.float32) #outputs is the shape of len(outputs)=num_step outputs[-1].get_shape()=(num_batch,num_hidden) 
		# Returns:A pair (outputs, state) where: - outputs is a length T list of outputs (one for each input), or a nested tuple of such elements. - state is the final state
		result = tf.add(tf.matmul(outputs[-1],w['out']),b['out'])
		return result

predict_label = LSTM_model(input_feature_reshape,weights,biases)
with tf.name_scope('Cost'):
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict_label,input_label))
with tf.name_scope('Adam_Optimizer'):
	optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

with tf.name_scope('Accuracy'):	
	correct = tf.cast(tf.equal(tf.argmax(predict_label,1),tf.argmax(input_label,1)),tf.float32)
	correct_rate = tf.reduce_mean(correct)

tf.scalar_summary("Cost", cost)
tf.scalar_summary("Accuracy", correct_rate)
merged_summary = tf.merge_all_summaries()

summary_writer = tf.train.SummaryWriter(log_path, graph=tf.get_default_graph())

sess = tf.Session()
sess.run(tf.initialize_all_variables())
batch_total = mnist.train.num_examples//batch_size
for epoch in range(epochs):
	#cost_sum = 0.0
	for batch in range(batch_total):
		batch_feature , batch_label = mnist.train.next_batch(batch_size)
		_ , summary_receive = sess.run([optimizer, merged_summary], feed_dict={input_feature:batch_feature, input_label:batch_label})
		#cost_sum += cost_receive
		summary_writer.add_summary(summary_receive, epoch * batch_total + batch)
	#cost_avg = cost_sum/batch_total
	#print("Epoch ", epoch , " Cost : ", cost_avg, "Correct rate: ",cor_rate)
print("Training Done!")

correct_result = sess.run(correct_rate,feed_dict={input_feature:mnist.test.images, input_label:mnist.test.labels})
print("Test Accuracy : ", correct_result)
sess.close()
print("Done!")
print("Run the command line:\n" \
          "--> tensorboard --logdir=%s " \
          "\nThen open http://127.0.1.1:6006/ into your web browser" % log_path)