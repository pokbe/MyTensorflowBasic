import tensorflow as tf
import numpy as np
import random

class SequenceDataGeneration(object):
	def __init__(self, num_examples, max_seq_len, min_seq_len, max_value):
		self.num_examples = num_examples
		self.features = []
		self.labels = []
		self.lengths = []

		for example in range(num_examples):
			example_length = random.randint(min_seq_len,max_seq_len)  #Return a random integer N such that a <= N <= b
			self.lengths.append(example_length)
			if random.random()<0.5 : #Return the next random floating point number in the range [0.0, 1.0)
				start_rand = random.randint(0, max_value-example_length)
				sequence = [x/max_value for x in range(start_rand, start_rand+example_length)]
				sequence += [0.0 for _ in range(max_seq_len - example_length)]
				self.features.append(sequence)
				self.labels.append([1.0, 0.0])
			else:
				sequence = [ random.randint(0, max_value)/max_value for _ in range(example_length) ]
				sequence += [0.0 for _ in range(max_seq_len - example_length)]
				self.features.append(sequence)
				self.labels.append([0.0, 1.0])
		self.batch_index = 0
	def next_batch(self, batch_size):
		if self.batch_index = self.num_examples:
			self.batch_index = 0
		batch_features = self.features[self.batch_index : min(self.batch_index+batch_size , self.num_examples )]
		batch_labels = self.labels[self.batch_index : min(self.batch_index+batch_size , self.num_examples )]
		batch_lengths = self.lengths[self.batch_index : min(self.batch_index+batch_size , self.num_examples )]
		batch_index = min(self.batch_index+batch_size , self.num_examples )
		return batch_features, batch_labels, batch_lengths

max_len = 50
min_len = 5
max_value = 500

num_input = 1
num_step = max_len
num_hidden = 100
num_class = 2

input_features = tf.placeholder(tf.float32,[None,num_step*num_input])
input_feature_reshape = tf.reshape(input_features,[-1,num_step,num_input])
input_labels = tf.placeholder(tf.float32,[None,num_class])
input_lengths = tf.placeholder(tf.int32,[None])

weights = {
	'out' : tf.Variable(tf.random_normal([2*num_hidden,num_class]))
}
biases = {
	'out' : tf.Variable(tf.random_normal([num_class]))
}

def BiLSTM_model(input_raw, s_len, w, b):
	input_list = tf.unpack(input_raw, axis=1)

	fw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, forget_bias=0.9)
	bw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, forget_bias=0.9)

	outputs, output_state_fw, output_state_bw = tf.nn.bidirectional_rnn(fw_cell, bw_cell, input_list, sequence_length=s_len,dtype = tf.float32)
	result = tf.add(tf.matmul(outputs[-1], w['out']),b['out'])
	return result

predict_label = BiLSTM_model(input_feature_reshape, input_lengths, weights , biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict_label, input_labels))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

correct = tf.cast(tf.equal(tf.argmax(predict_label, 1), tf.argmax(input_labels, 1)) , tf.float32)
correct_rate = tf.reduce_mean(correct)

train_data = SequenceDataGeneration(num_examples = 1000, max_seq_len=max_len, min_seq_len=min_len, max_value=max_value)
test_data = SequenceDataGeneration(num_examples = 500, max_seq_len=max_len, min_seq_len=min_len, max_value=max_value)

epochs = 10
batch_size = 50

sess = tf.Session()
sess.run(tf.initialize_all_variables())
batch_total = train_data.num_examples//batch_size
for epoch in range(epochs):
	cost_sum = 0.0
	for batch in range(batch_total):
		batch_features, batch_labels, batch_lengths = train_data.next_batch(batch_size)
		_, cost_receive, cor_rate = sess.run([optimizer,cost,correct_rate],feed_dict={input_features:batch_features, input_labels:batch_labels, input_lengths:batch_lengths})
		cost_sum += cost_receive
	cost_avg = cost_sum/batch_total
	print("Epoch ", epoch , " Cost : ", cost_avg, "Correct rate: ",cor_rate)
print("Training Done !!!")

correct_result = sess.run(correct_rate, feed_dict={input_features:test_data.features, input_labels:test_data.labels, input_lengths:test_data.lengths})
print("Test Accuracy : ", correct_result)
sess.close()

print("Done !!")