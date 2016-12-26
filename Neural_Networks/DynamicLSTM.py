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
		if self.batch_index == self.num_examples:
			self.batch_index = 0
		batch_features = self.features[self.batch_index : min(self.batch_index+batch_size , self.num_examples )]
		batch_labels = self.labels[self.batch_index : min(self.batch_index+batch_size , self.num_examples )]
		batch_lengths = self.lengths[self.batch_index : min(self.batch_index+batch_size , self.num_examples )]
		self.batch_index = min(self.batch_index+batch_size , self.num_examples )
		return batch_features, batch_labels, batch_lengths

max_len = 40
min_len = 10
max_value = 500

num_input = 1
num_step = max_len
num_hidden = 10
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
	outputs = tf.pack(outputs)
	print(outputs.get_shape()) # (num_step, batch_size, 2*num_hidden)
	outputs = tf.transpose(outputs, [1, 0, 2]) # (batch_size, num_step, 2*num_hidden)
	#print(outputs.get_shape())
	batch_size_get = tf.shape(outputs)[0]
	#print(batch_size_get)
	index = tf.range(0, batch_size_get)*max_len + (s_len - 1)
	#print(index.get_shape()) #(batch_size,)
	outputs_true = tf.gather(tf.reshape(outputs, [-1, 2*num_hidden]), index)
	#print(outputs_true.get_shape()) #(batch_size, 20)
	result = tf.add(tf.matmul(outputs_true, w['out']),b['out'])
	return result

predict_label = BiLSTM_model(input_feature_reshape, input_lengths, weights , biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict_label, input_labels))
optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

correct = tf.cast(tf.equal(tf.argmax(predict_label, 1), tf.argmax(input_labels, 1)) , tf.float32)
correct_rate = tf.reduce_mean(correct)

train_data = SequenceDataGeneration(num_examples = 1000, max_seq_len=max_len, min_seq_len=min_len, max_value=max_value)
test_data = SequenceDataGeneration(num_examples = 500, max_seq_len=max_len, min_seq_len=min_len, max_value=max_value)

epochs = 100
batch_size = 50

sess = tf.Session()
sess.run(tf.initialize_all_variables())
batch_total = train_data.num_examples//batch_size
for epoch in range(epochs):
	cost_sum = 0.0
	for batch in range(batch_total):
		batch_features, batch_labels, batch_lengths = train_data.next_batch(batch_size)
		#o = sess.run(oo,feed_dict={input_features:batch_features, input_labels:batch_labels, input_lengths:batch_lengths})
		#print(o)
		_, cost_receive, cor_rate = sess.run([optimizer,cost,correct_rate],feed_dict={input_features:batch_features, input_labels:batch_labels, input_lengths:batch_lengths})
		cost_sum += cost_receive
		#print(cor_rate)
	cost_avg = cost_sum/batch_total
	print("Epoch ", epoch , " Cost : ", cost_avg, "Correct rate: ",cor_rate)
print("Training Done !!!")

correct_result = sess.run(correct_rate, feed_dict={input_features:test_data.features, input_labels:test_data.labels, input_lengths:test_data.lengths})
print("Test Accuracy : ", correct_result)
sess.close()

print("Done !!")
'''
Epoch  1  Cost :  0.641525068879 Correct rate:  0.8
Epoch  2  Cost :  0.504662966728 Correct rate:  0.78
Epoch  3  Cost :  0.427366574109 Correct rate:  0.92
Epoch  4  Cost :  0.416661828756 Correct rate:  0.88
Epoch  5  Cost :  0.35026936233 Correct rate:  0.94
Epoch  6  Cost :  0.294951407611 Correct rate:  0.92
Epoch  7  Cost :  0.321931604296 Correct rate:  0.94
Epoch  8  Cost :  0.258613362163 Correct rate:  0.86
Epoch  9  Cost :  0.16586479526 Correct rate:  0.98
Epoch  10  Cost :  0.721936706454 Correct rate:  0.58
Epoch  11  Cost :  0.411165941507 Correct rate:  0.96
Epoch  12  Cost :  0.214789842442 Correct rate:  0.96
Epoch  13  Cost :  0.14348605983 Correct rate:  0.96
Epoch  14  Cost :  0.10064470116 Correct rate:  0.98
Epoch  15  Cost :  0.109889965504 Correct rate:  0.98
Epoch  16  Cost :  0.0633457833901 Correct rate:  1.0
Epoch  17  Cost :  0.0448067913763 Correct rate:  1.0
Epoch  18  Cost :  0.114841242274 Correct rate:  0.88
Epoch  19  Cost :  0.202483319212 Correct rate:  0.98
Epoch  20  Cost :  0.0897681605071 Correct rate:  0.98
Epoch  21  Cost :  0.0527777417563 Correct rate:  1.0
Epoch  22  Cost :  0.0387155518867 Correct rate:  1.0
Epoch  23  Cost :  0.027088847151 Correct rate:  1.0
Epoch  24  Cost :  0.0233905521687 Correct rate:  1.0
Epoch  25  Cost :  0.0226432107389 Correct rate:  1.0
Epoch  26  Cost :  0.0224349411088 Correct rate:  1.0
Epoch  27  Cost :  0.0219253485207 Correct rate:  1.0
Epoch  28  Cost :  0.0216249423916 Correct rate:  1.0
Epoch  29  Cost :  0.0188906583237 Correct rate:  1.0
Epoch  30  Cost :  0.0127528410405 Correct rate:  1.0
Epoch  31  Cost :  0.00894623131026 Correct rate:  1.0
Epoch  32  Cost :  0.00643337656511 Correct rate:  1.0
Epoch  33  Cost :  0.00528640524717 Correct rate:  1.0
Epoch  34  Cost :  0.00469252310577 Correct rate:  1.0
Epoch  35  Cost :  0.00415027205308 Correct rate:  1.0
Epoch  36  Cost :  0.00368625283299 Correct rate:  1.0
Epoch  37  Cost :  0.0033413031837 Correct rate:  1.0
Epoch  38  Cost :  0.00306728118157 Correct rate:  1.0
Epoch  39  Cost :  0.00282837069244 Correct rate:  1.0
Epoch  40  Cost :  0.00262613428349 Correct rate:  1.0
Epoch  41  Cost :  0.00245682317181 Correct rate:  1.0
Epoch  42  Cost :  0.00230976889579 Correct rate:  1.0
Epoch  43  Cost :  0.00218075314042 Correct rate:  1.0
Epoch  44  Cost :  0.00206780346198 Correct rate:  1.0
Epoch  45  Cost :  0.00196783232968 Correct rate:  1.0
Epoch  46  Cost :  0.00187853512907 Correct rate:  1.0
Epoch  47  Cost :  0.00179843572405 Correct rate:  1.0
Epoch  48  Cost :  0.00172616226555 Correct rate:  1.0
Epoch  49  Cost :  0.00166056714515 Correct rate:  1.0
Epoch  50  Cost :  0.00160077586552 Correct rate:  1.0
Epoch  51  Cost :  0.00154602415714 Correct rate:  1.0
Epoch  52  Cost :  0.00149566584005 Correct rate:  1.0
Epoch  53  Cost :  0.00144917940343 Correct rate:  1.0
Epoch  54  Cost :  0.00140610693998 Correct rate:  1.0
Epoch  55  Cost :  0.00136604105282 Correct rate:  1.0
Epoch  56  Cost :  0.00132865522246 Correct rate:  1.0
Epoch  57  Cost :  0.00129365345856 Correct rate:  1.0
Epoch  58  Cost :  0.00126078137037 Correct rate:  1.0
Epoch  59  Cost :  0.00122981877939 Correct rate:  1.0
Epoch  60  Cost :  0.00120055424777 Correct rate:  1.0
Epoch  61  Cost :  0.00117282263927 Correct rate:  1.0
Epoch  62  Cost :  0.00114649141033 Correct rate:  1.0
Epoch  63  Cost :  0.00112140874917 Correct rate:  1.0
Epoch  64  Cost :  0.00109746736853 Correct rate:  1.0
Epoch  65  Cost :  0.0010745678992 Correct rate:  1.0
Epoch  66  Cost :  0.00105262731886 Correct rate:  1.0
Epoch  67  Cost :  0.00103154805838 Correct rate:  1.0
Epoch  68  Cost :  0.00101127597518 Correct rate:  1.0
Epoch  69  Cost :  0.000991751895708 Correct rate:  1.0
Epoch  70  Cost :  0.000972918202024 Correct rate:  1.0
Epoch  71  Cost :  0.000954730507146 Correct rate:  1.0
Epoch  72  Cost :  0.000937131450337 Correct rate:  1.0
Epoch  73  Cost :  0.000920096841219 Correct rate:  1.0
Epoch  74  Cost :  0.000903591883616 Correct rate:  1.0
Epoch  75  Cost :  0.000887576903551 Correct rate:  1.0
Epoch  76  Cost :  0.000872031345716 Correct rate:  1.0
Epoch  77  Cost :  0.000856926067718 Correct rate:  1.0
Epoch  78  Cost :  0.000842240893144 Correct rate:  1.0
Epoch  79  Cost :  0.000827935163034 Correct rate:  1.0
Epoch  80  Cost :  0.000813994806049 Correct rate:  1.0
Epoch  81  Cost :  0.000800412834542 Correct rate:  1.0
Epoch  82  Cost :  0.000787169346404 Correct rate:  1.0
Epoch  83  Cost :  0.000774225452915 Correct rate:  1.0
Epoch  84  Cost :  0.000761575141223 Correct rate:  1.0
Epoch  85  Cost :  0.000749216939585 Correct rate:  1.0
Epoch  86  Cost :  0.000737126156127 Correct rate:  1.0
Epoch  87  Cost :  0.000725284782675 Correct rate:  1.0
Epoch  88  Cost :  0.00071366467273 Correct rate:  1.0
Epoch  89  Cost :  0.000702256033946 Correct rate:  1.0
Epoch  90  Cost :  0.000691069923414 Correct rate:  1.0
Epoch  91  Cost :  0.000680094860218 Correct rate:  1.0
Epoch  92  Cost :  0.000669310480771 Correct rate:  1.0
Epoch  93  Cost :  0.000658693054447 Correct rate:  1.0
Epoch  94  Cost :  0.000648234623077 Correct rate:  1.0
Epoch  95  Cost :  0.000637931262827 Correct rate:  1.0
Epoch  96  Cost :  0.000627767559672 Correct rate:  1.0
Epoch  97  Cost :  0.000617741972928 Correct rate:  1.0
Epoch  98  Cost :  0.000607831940124 Correct rate:  1.0
Epoch  99  Cost :  0.000598033949063 Correct rate:  1.0
Training Done !!!
Test Accuracy :  1.0
Done !!
'''