import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

epochs = 15
batch_size = 100

num_input = 28
num_step = 28
num_hidden = 100
num_class = 10

input_feature = tf.placeholder(tf.float32,[None,num_step*num_input])
input_feature_reshape = tf.reshape(input_feature,[-1,num_step,num_input])
input_label = tf.placeholder(tf.float32,[None,num_class])
input_keep = tf.placeholder(tf.float32)

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
	result = tf.add(tf.matmul(outputs[-1], w['out']),b['out'])
	result_dropout = tf.nn.dropout(result,keep_prob=input_keep)
	return result_dropout

predict_label = BiLSTM_model(input_feature_reshape, weights , biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict_label, input_label))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

correct = tf.cast(tf.equal(tf.argmax(predict_label, 1), tf.argmax(input_label, 1)) , tf.float32)
correct_rate = tf.reduce_mean(correct)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
batch_total = mnist.train.num_examples//batch_size
for epoch in range(epochs):
	cost_sum = 0.0
	for batch in range(batch_total):
		batch_feature,batch_label = mnist.train.next_batch(batch_size)
		_, cost_receive, cor_rate = sess.run([optimizer,cost,correct_rate],feed_dict={input_feature:batch_feature, input_label:batch_label, input_keep:1.0})
		cost_sum += cost_receive
	cost_avg = cost_sum/batch_total
	print("Epoch ", epoch , " Cost : ", cost_avg, "Correct rate: ",cor_rate)
print("Training Done !!!")

correct_result = sess.run(correct_rate, feed_dict={input_feature:mnist.test.images, input_label:mnist.test.labels, input_keep:1.0})
print("Test Accuracy : ", correct_result)
sess.close()

print("Done !!")
#Without dropout
'''
Epoch  0  Cost :  0.51099120344 Correct rate:  0.97
Epoch  1  Cost :  0.134032688154 Correct rate:  0.98
Epoch  2  Cost :  0.0934430588562 Correct rate:  0.95
Epoch  3  Cost :  0.0738246156868 Correct rate:  0.97
Epoch  4  Cost :  0.0587947279062 Correct rate:  1.0
Epoch  5  Cost :  0.0502512688287 Correct rate:  0.99
Epoch  6  Cost :  0.0437438815318 Correct rate:  0.96
Epoch  7  Cost :  0.0375639037089 Correct rate:  0.99
Epoch  8  Cost :  0.0331736529331 Correct rate:  1.0
Epoch  9  Cost :  0.029384205348 Correct rate:  0.97
Epoch  10  Cost :  0.0284328267168 Correct rate:  0.99
Epoch  11  Cost :  0.023100302018 Correct rate:  1.0
Epoch  12  Cost :  0.0228666330237 Correct rate:  1.0
Epoch  13  Cost :  0.0195257928364 Correct rate:  1.0
Epoch  14  Cost :  0.0188068480342 Correct rate:  1.0
Training Done !!!
Test Accuracy :  0.988
Done !!
'''
#Dropout
'''
Epoch  0  Cost :  0.860979960236 Correct rate:  0.78
Epoch  1  Cost :  0.538261290138 Correct rate:  0.79
Epoch  2  Cost :  0.483944747123 Correct rate:  0.78
Epoch  3  Cost :  0.464167511463 Correct rate:  0.77
Epoch  4  Cost :  0.447733883858 Correct rate:  0.82
Epoch  5  Cost :  0.435608209426 Correct rate:  0.77
Epoch  6  Cost :  0.423532867134 Correct rate:  0.8
Epoch  7  Cost :  0.418949091489 Correct rate:  0.79
Epoch  8  Cost :  0.41603947271 Correct rate:  0.82
Epoch  9  Cost :  0.413360322524 Correct rate:  0.84
Epoch  10  Cost :  0.409654457244 Correct rate:  0.72
Epoch  11  Cost :  0.402498143722 Correct rate:  0.78
Epoch  12  Cost :  0.404318758574 Correct rate:  0.82
Epoch  13  Cost :  0.399037350037 Correct rate:  0.87
Epoch  14  Cost :  0.399590348656 Correct rate:  0.8
Training Done !!!
Test Accuracy :  0.985
Done !!
'''