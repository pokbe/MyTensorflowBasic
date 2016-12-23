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
def out_layer(input, weights, biases):
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
'''
Epoch  0  : Cost-  5309.7343727 Correct-  0.89
Epoch  1  : Cost-  583.507481658 Correct-  0.92
Epoch  2  : Cost-  307.636656366 Correct-  0.96
Epoch  3  : Cost-  178.921099162 Correct-  0.97
Epoch  4  : Cost-  129.128687656 Correct-  0.96
Epoch  5  : Cost-  95.4323399412 Correct-  0.93
Epoch  6  : Cost-  74.8586167776 Correct-  0.93
Epoch  7  : Cost-  61.4043292473 Correct-  0.98
Epoch  8  : Cost-  50.1631639219 Correct-  0.95
Epoch  9  : Cost-  42.0938581476 Correct-  0.97
Epoch  10  : Cost-  36.7214751951 Correct-  0.91
Epoch  11  : Cost-  30.0273034518 Correct-  0.97
Epoch  12  : Cost-  26.17893616 Correct-  0.98
Epoch  13  : Cost-  22.4949802409 Correct-  0.93
Epoch  14  : Cost-  20.3499195997 Correct-  0.97
Epoch  15  : Cost-  17.077230375 Correct-  0.96
Epoch  16  : Cost-  14.4293103978 Correct-  0.96
Epoch  17  : Cost-  13.9672610943 Correct-  0.97
Epoch  18  : Cost-  11.3547397702 Correct-  1.0
Epoch  19  : Cost-  11.8605678545 Correct-  0.98
Epoch  20  : Cost-  10.41934072 Correct-  1.0
Epoch  21  : Cost-  9.71769613547 Correct-  1.0
Epoch  22  : Cost-  9.98920694053 Correct-  0.98
Epoch  23  : Cost-  8.03916387691 Correct-  0.98
Epoch  24  : Cost-  7.35626363375 Correct-  0.98
Epoch  25  : Cost-  6.69535694934 Correct-  0.99
Epoch  26  : Cost-  6.69817827347 Correct-  0.98
Epoch  27  : Cost-  5.89217425451 Correct-  0.98
Epoch  28  : Cost-  6.87720565434 Correct-  0.98
Epoch  29  : Cost-  5.35962987068 Correct-  0.99
Epoch  30  : Cost-  4.26914015073 Correct-  1.0
Epoch  31  : Cost-  4.19181674527 Correct-  1.0
Epoch  32  : Cost-  4.38480315666 Correct-  0.99
Epoch  33  : Cost-  3.9839688209 Correct-  1.0
Epoch  34  : Cost-  4.07533097178 Correct-  1.0
Epoch  35  : Cost-  4.16567884677 Correct-  0.98
Epoch  36  : Cost-  3.47000602092 Correct-  0.99
Epoch  37  : Cost-  3.17121943173 Correct-  0.99
Epoch  38  : Cost-  3.06483391337 Correct-  0.99
Epoch  39  : Cost-  3.20044917494 Correct-  0.99
Epoch  40  : Cost-  2.70981469526 Correct-  0.99
Epoch  41  : Cost-  3.05142584564 Correct-  1.0
Epoch  42  : Cost-  3.01105321701 Correct-  1.0
Epoch  43  : Cost-  2.51552708847 Correct-  0.99
Epoch  44  : Cost-  2.18140772109 Correct-  1.0
Epoch  45  : Cost-  2.19475024334 Correct-  1.0
Epoch  46  : Cost-  2.08887576354 Correct-  1.0
Epoch  47  : Cost-  2.10661054158 Correct-  1.0
Epoch  48  : Cost-  2.57771294137 Correct-  0.99
Epoch  49  : Cost-  2.38656898838 Correct-  1.0
Epoch  50  : Cost-  1.83856851424 Correct-  1.0
Epoch  51  : Cost-  1.88658466412 Correct-  1.0
Epoch  52  : Cost-  1.74147498498 Correct-  1.0
Epoch  53  : Cost-  1.89783069354 Correct-  0.97
Epoch  54  : Cost-  1.85689779301 Correct-  1.0
Epoch  55  : Cost-  1.91342573278 Correct-  1.0
Epoch  56  : Cost-  1.23836577519 Correct-  0.99
Epoch  57  : Cost-  1.45605462002 Correct-  0.99
Epoch  58  : Cost-  1.54806495179 Correct-  0.98
Epoch  59  : Cost-  1.57440927392 Correct-  1.0
Epoch  60  : Cost-  1.58429213726 Correct-  0.99
Epoch  61  : Cost-  1.43342467349 Correct-  1.0
Epoch  62  : Cost-  1.35137190229 Correct-  0.98
Epoch  63  : Cost-  1.75398027144 Correct-  1.0
Epoch  64  : Cost-  1.62915508753 Correct-  1.0
Epoch  65  : Cost-  1.37695998689 Correct-  1.0
Epoch  66  : Cost-  1.37100960592 Correct-  1.0
Epoch  67  : Cost-  1.18502589711 Correct-  0.99
Epoch  68  : Cost-  1.49229806108 Correct-  1.0
Epoch  69  : Cost-  1.38533186183 Correct-  1.0
Epoch  70  : Cost-  1.21121021902 Correct-  1.0
Epoch  71  : Cost-  1.04159412765 Correct-  0.98
Epoch  72  : Cost-  1.15168505976 Correct-  1.0
Epoch  73  : Cost-  0.990220375722 Correct-  0.97
Epoch  74  : Cost-  1.245360281 Correct-  1.0
Epoch  75  : Cost-  1.02536057379 Correct-  1.0
Epoch  76  : Cost-  1.23585335929 Correct-  1.0
Epoch  77  : Cost-  0.938250958963 Correct-  1.0
Epoch  78  : Cost-  1.23122870353 Correct-  0.99
Epoch  79  : Cost-  1.11359851 Correct-  1.0
Epoch  80  : Cost-  1.01177757653 Correct-  1.0
Epoch  81  : Cost-  0.7917736034 Correct-  1.0
Epoch  82  : Cost-  1.19520129482 Correct-  1.0
Epoch  83  : Cost-  1.13174477053 Correct-  1.0
Epoch  84  : Cost-  1.08375113068 Correct-  1.0
Epoch  85  : Cost-  0.912650929475 Correct-  1.0
Epoch  86  : Cost-  0.876019384155 Correct-  1.0
Epoch  87  : Cost-  0.873602084874 Correct-  0.99
Epoch  88  : Cost-  1.00426976275 Correct-  0.99
Epoch  89  : Cost-  1.09992056411 Correct-  1.0
Epoch  90  : Cost-  0.984228480065 Correct-  0.99
Epoch  91  : Cost-  0.547688289509 Correct-  1.0
Epoch  92  : Cost-  0.898461572871 Correct-  0.99
Epoch  93  : Cost-  1.33369234353 Correct-  1.0
Epoch  94  : Cost-  0.783234222188 Correct-  0.99
Epoch  95  : Cost-  0.833581909103 Correct-  1.0
Epoch  96  : Cost-  0.765362604915 Correct-  1.0
Epoch  97  : Cost-  0.73851467397 Correct-  1.0
Epoch  98  : Cost-  0.809658869408 Correct-  1.0
Epoch  99  : Cost-  0.805113238332 Correct-  1.0
Training Done!
Test correct rate :  0.9897
Done
'''