import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

epochs = 50
batch_size = 100

num_feature = 784
num_class = 10
num_hidden_1 = 300
num_hidden_2 = 300

model_path = 'model/muti_perception_model.ckpt'

input_feature = tf.placeholder(tf.float32, [None,num_feature] )
input_label = tf.placeholder(tf.float32, [None, num_class])

weights = {
	'hidden_1' : tf.Variable(tf.random_normal([num_feature,num_hidden_1])),
	'hidden_2' : tf.Variable(tf.random_normal([num_hidden_1,num_hidden_2])),
	'output' : tf.Variable(tf.random_normal([num_hidden_2,num_class]))
}
biases = {
	'hidden_1' : tf.Variable(tf.random_normal([num_hidden_1])),
	'hidden_2' : tf.Variable(tf.random_normal([num_hidden_2])),
	'output' : tf.Variable(tf.random_normal([num_class]))
}

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

saver = tf.train.Saver()

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

save_path = saver.save(sess, model_path)
print("Save path : ", save_path)

sess.close()

'''
Epoch  0  Cost :  57.5065751068
Epoch  1  Cost :  5.95275713685
Epoch  2  Cost :  2.83561577108
Epoch  3  Cost :  1.86087028512
Epoch  4  Cost :  1.19503725366
Epoch  5  Cost :  0.853646354336
Epoch  6  Cost :  0.607775726768
Epoch  7  Cost :  0.470330512233
Epoch  8  Cost :  0.383522053401
Epoch  9  Cost :  0.300153976426
Epoch  10  Cost :  0.284897794374
Epoch  11  Cost :  0.351505755538
Epoch  12  Cost :  0.308314197219
Epoch  13  Cost :  0.248429064998
Epoch  14  Cost :  0.219342248819
Epoch  15  Cost :  0.22209554016
Epoch  16  Cost :  0.188208203814
Epoch  17  Cost :  0.205742311898
Epoch  18  Cost :  0.207301017569
Epoch  19  Cost :  0.27270909505
Epoch  20  Cost :  0.215638749935
Epoch  21  Cost :  0.307883285819
Epoch  22  Cost :  0.241591632291
Epoch  23  Cost :  0.268113160093
Epoch  24  Cost :  0.279211948108
Epoch  25  Cost :  0.251916283024
Epoch  26  Cost :  0.25193314469
Epoch  27  Cost :  0.405640056655
Epoch  28  Cost :  0.347810693187
Epoch  29  Cost :  0.330608272261
Epoch  30  Cost :  0.231939708834
Epoch  31  Cost :  0.310512739833
Epoch  32  Cost :  0.307011644678
Epoch  33  Cost :  0.308260595104
Epoch  34  Cost :  0.322802134441
Epoch  35  Cost :  0.260776191618
Epoch  36  Cost :  0.294695831381
Epoch  37  Cost :  0.233169394931
Epoch  38  Cost :  0.294241600558
Epoch  39  Cost :  0.288422233379
Epoch  40  Cost :  0.327317817469
Epoch  41  Cost :  0.308031905192
Epoch  42  Cost :  0.363491062346
Epoch  43  Cost :  0.306587231058
Epoch  44  Cost :  0.337259377377
Epoch  45  Cost :  0.357900326509
Epoch  46  Cost :  0.397240722315
Epoch  47  Cost :  0.350347452719
Epoch  48  Cost :  0.367660887973
Epoch  49  Cost :  0.314137046893
Test Accuracy :  0.919
Save path :  model/muti_perception_model.ckpt
'''