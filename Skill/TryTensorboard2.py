import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

epochs = 20
batch_size = 100

num_feature = 784
num_class = 10
num_hidden_1 = 300
num_hidden_2 = 300

log_path = 'mp_log/1/'

weights = {
	'hidden_1' : tf.Variable(tf.random_normal([num_feature,num_hidden_1]), name = 'weights_1'),
	'hidden_2' : tf.Variable(tf.random_normal([num_hidden_1,num_hidden_2]), name = 'weights_2'),
	'output' : tf.Variable(tf.random_normal([num_hidden_2,num_class]), name = 'weights_3')
}
biases = {
	'hidden_1' : tf.Variable(tf.random_normal([num_hidden_1]), name='biases_1'),
	'hidden_2' : tf.Variable(tf.random_normal([num_hidden_2]), name='biases_2'),
	'output' : tf.Variable(tf.random_normal([num_class]), name='biases_3')
}

input_feature = tf.placeholder(tf.float32,[None,num_feature], name= 'Input_Feature')
input_label = tf.placeholder(tf.float32,[None,num_class], name= 'Input_Label')

def mulilayerperception(input,weights,biases):
	hidden_layer_1 = tf.nn.relu(tf.add(tf.matmul(input,weights['hidden_1']),biases['hidden_1']))
	tf.histogram_summary("hidden_layer_1_relu", hidden_layer_1)
	hidden_layer_2 = tf.nn.relu(tf.add(tf.matmul(hidden_layer_1,weights['hidden_2']),biases['hidden_2']))
	tf.histogram_summary("hidden_layer_2_relu", hidden_layer_2)
	output_layer = tf.add(tf.matmul(hidden_layer_2,weights['output']),biases['output'])
	return output_layer

with tf.name_scope('MP_model'):
	predict_label = mulilayerperception(input_feature,weights,biases)
with tf.name_scope('Cost'):
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict_label,input_label))
with tf.name_scope('Optimizer'):
	optimizer = tf.train.GradientDescentOptimizer(0.02)
	# Op to calculate every variable gradient
	grads = tf.gradients(cost, tf.trainable_variables())
	grads = list(zip(grads, tf.trainable_variables()))
	# Op to update all variables according to their gradient
	apply_grads = optimizer.apply_gradients(grads_and_vars=grads)
with tf.name_scope('Accuracy'):
	correct = tf.cast(tf.equal(tf.argmax(predict_label,1),tf.argmax(input_label,1)),tf.float32)
	correct_rate = tf.reduce_mean(correct)

tf.scalar_summary("Cost", cost)
tf.scalar_summary("Accuracy", correct_rate)
for var in tf.trainable_variables():
	tf.histogram_summary(var.name, var)
for grad, var in grads:
	tf.histogram_summary(var.name + '/gradient', grad)

merged_summary = tf.merge_all_summaries()

summary_writer = tf.train.SummaryWriter(log_path, graph=tf.get_default_graph())

sess = tf.Session()
sess.run(tf.initialize_all_variables())

batch_total = mnist.train.num_examples//batch_size
for epoch in range(epochs):
	cost_sum = 0.0
	for batch in range(batch_total):
		batch_feature , batch_label = mnist.train.next_batch(batch_size)
		_ , cost_receive, summary_receive = sess.run([apply_grads, cost, merged_summary], feed_dict={input_feature:batch_feature, input_label:batch_label})
		cost_sum += cost_receive
		summary_writer.add_summary(summary_receive, epoch * batch_total + batch)
	cost_avg = cost_sum/batch_total
	print("Epoch ", epoch , " Cost : ", cost_avg)

correct_result = sess.run(correct_rate,feed_dict={input_feature:mnist.test.images, input_label:mnist.test.labels})
print("Test Accuracy : ", correct_result)

sess.close()
print("Done!")
print("Run the command line:\n" \
          "--> tensorboard --logdir=%s " \
          "\nThen open http://127.0.1.1:6006/ into your web browser" % log_path)
'''
Epoch  0  Cost :  56.9205420854
Epoch  1  Cost :  7.01636041748
Epoch  2  Cost :  3.84542901023
Epoch  3  Cost :  2.53874731673
Epoch  4  Cost :  1.83229702714
Epoch  5  Cost :  1.40066637032
Epoch  6  Cost :  1.09038134382
Epoch  7  Cost :  0.878553340621
Epoch  8  Cost :  0.737845001272
Epoch  9  Cost :  0.633460803991
Epoch  10  Cost :  0.537754767722
Epoch  11  Cost :  0.461938573706
Epoch  12  Cost :  0.401673750581
Epoch  13  Cost :  0.345189492648
Epoch  14  Cost :  0.317771050137
Epoch  15  Cost :  0.282020796079
Epoch  16  Cost :  0.252921797883
Epoch  17  Cost :  0.229976943784
Epoch  18  Cost :  0.206952842599
Epoch  19  Cost :  0.192591045256
Test Accuracy :  0.9145
Done!
Run the command line:
--> tensorboard --logdir=mp_log/1/ 
Then open http://127.0.1.1:6006/ into your web browser
'''