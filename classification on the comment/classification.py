import nltk
import numpy as np
import random
import tensorflow as tf

sent = "He is a Chinese actress"

from nltk.tokenize import word_tokenize

sent_token = word_tokenize(sent)
print(sent_token)
"""
['He', 'is', 'a', 'Chinese', 'actress']
"""

from nltk.stem import WordNetLemmatizer #http://www.nltk.org/book/ch03.html
sent = "He said he has been a member of teachers."
sent_token = word_tokenize(sent)
wn_lemmatizer = WordNetLemmatizer()
sent_origin = [wn_lemmatizer.lemmatize(token) for token in sent_token]
print(sent_origin)
"""
['He', 'said', 'he', 'ha', 'been', 'a', 'member', 'of', 'teacher', '.']
"""

pos_file = './positive_comment.txt'
neg_file = './negative_comment.txt'

from collections import Counter

def build_vocabulary(pos_file,neg_file):
	def file2wordlist(filepath):
		wn_lemmatizer = WordNetLemmatizer()
		wordlist = []
		f = open(filepath,"r")
		for line in f.readlines():
			words = word_tokenize(line.lower())
			words = [wn_lemmatizer.lemmatize(word) for word in words]
			wordlist.extend(words)
		f.close()
		return wordlist
	wordlist = []
	wordlist.extend(file2wordlist(pos_file))
	wordlist.extend(file2wordlist(neg_file))

	word_counter = Counter(wordlist)
	vocabulary = []
	for word in word_counter:
		if word_counter[word]<2000 and word_counter[word]>20:
			vocabulary.append(word)
	print("Build Vocabulary Length: ", len(vocabulary)) #Build Vocabulary Length:  1063
	return vocabulary

vocabulary = build_vocabulary(pos_file,neg_file)
#print(vocabulary)
def string2vec(line,vocabulary,label):
	wn_lemmatizer = WordNetLemmatizer()
	words = word_tokenize(line.lower())
	words = [wn_lemmatizer.lemmatize(word) for word in words]
	features = np.zeros(len(vocabulary))
	for word in words:
		if word in vocabulary:
			index = vocabulary.index(word)
			features[index] += 1
	return [features,label]

def review2vec(pos_file,neg_file,vocabulary):
	dataset = []

	posf = open(pos_file,"r")
	for line in posf.readlines():
		sample_vec = string2vec(line,vocabulary,[1,0])
		dataset.append(sample_vec)
	posf.close()
	negf = open(neg_file,"r")
	for line in negf.readlines():
		sample_vec = string2vec(line,vocabulary,[0,1])
		dataset.append(sample_vec)
	negf.close()

	return dataset

dataset = review2vec(pos_file=pos_file,neg_file=neg_file,vocabulary=vocabulary)
#print(dataset)
random.shuffle(dataset)

test_rate = 0.3
batch_size = 30
epochs = 30

dataset = np.array(dataset)
#print(dataset)
'''
[[array([ 0.,  0.,  0., ...,  0.,  0.,  0.]) [0, 1]]
 [array([ 0.,  0.,  0., ...,  0.,  0.,  0.]) [0, 1]]
 [array([ 0.,  0.,  0., ...,  0.,  0.,  0.]) [1, 0]]
 ..., 
 [array([ 0.,  0.,  0., ...,  0.,  0.,  0.]) [0, 1]]
 [array([ 0.,  0.,  0., ...,  0.,  0.,  0.]) [0, 1]]
 [array([ 0.,  0.,  0., ...,  0.,  0.,  0.]) [0, 1]]]
'''
test_length = int(test_rate*len(dataset))
test_data = dataset[0: test_length]
train_data = dataset[test_length:-1]

input_size = len(vocabulary)
hidden_size_1 = 500
hidden_size_2 = 100
output_size = 2

weights = {
	"hidden_1" : tf.Variable(tf.random_normal([input_size,hidden_size_1])),
	"hidden_2" : tf.Variable(tf.random_normal([hidden_size_1,hidden_size_2])),
	"output" : tf.Variable(tf.random_normal([hidden_size_2,output_size]))
}
biases = {
	"hidden_1" : tf.Variable(tf.random_normal([hidden_size_1])),
	"hidden_2" : tf.Variable(tf.random_normal([hidden_size_2])),
	"output" : tf.Variable(tf.random_normal([output_size]))
}

input_feature = tf.placeholder(tf.float32,[None,input_size])
input_label = tf.placeholder(tf.float32,[None,2])

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

batch_total = len(train_data)//batch_size

for epoch in range(epochs):
	random.shuffle(train_data)
	cost_sum = 0.0
	for batch in range(batch_total):
		batch_data = train_data[batch*batch_size:(batch+1)*batch_size]
		batch_feature = np.array(batch_data[:,0])
		batch_label = np.array(batch_data[:,1])
		#print(type(batch_feature)) #<class 'numpy.ndarray'>
		#print(batch_feature.shape) # (30,)
		#print(type(batch_feature[0])) # <class 'numpy.ndarray'>
		#print(batch_feature[0].shape) # (1063,)
		#print(type(batch_label)) # <class 'numpy.ndarray'>
		#print(batch_label.shape) #(30,)
		#print(type(batch_label[0])) # <class 'list'>
		#print(len(batch_label[0])) # 2
		_ , cost_receive = sess.run([optimizer, cost], feed_dict={
				input_feature:list(batch_feature), 
				input_label:list(batch_label)
				})
		cost_sum += cost_receive
	cost_avg = cost_sum/batch_total
	print("Epoch ", epoch , " Cost : ", cost_avg)

correct_result = sess.run(correct_rate,feed_dict={
		input_feature:list(test_data[:,0]), 
		input_label:list(test_data[:,1])
		})
print("Test Accuracy : ", correct_result)

sess.close()

'''
Epoch  0  Cost :  20.7291155346
Epoch  1  Cost :  0.309244243789
Epoch  2  Cost :  0.11282623102
Epoch  3  Cost :  0.0604823797043
Epoch  4  Cost :  0.0439650178732
Epoch  5  Cost :  0.0346152627274
Epoch  6  Cost :  0.0221023704333
Epoch  7  Cost :  0.0127257362007
Epoch  8  Cost :  0.00712184218778
Epoch  9  Cost :  0.00326008539236
Epoch  10  Cost :  0.00070831882133
Epoch  11  Cost :  0.000294918416479
Epoch  12  Cost :  4.13290576288e-05
Epoch  13  Cost :  1.37632589205e-05
Epoch  14  Cost :  2.21049898844e-07
Epoch  15  Cost :  1.19497691111e-07
Epoch  16  Cost :  1.18824735972e-07
Epoch  17  Cost :  1.18968940907e-07
Epoch  18  Cost :  1.19097123103e-07
Epoch  19  Cost :  1.19193259678e-07
Epoch  20  Cost :  1.19209282445e-07
Epoch  21  Cost :  1.19209282445e-07
Epoch  22  Cost :  1.19209282445e-07
Epoch  23  Cost :  1.19209282445e-07
Epoch  24  Cost :  1.19209282445e-07
Epoch  25  Cost :  3.22057335639e-08
Epoch  26  Cost :  0.0
Epoch  27  Cost :  0.0
Epoch  28  Cost :  0.0
Epoch  29  Cost :  0.0
Test Accuracy :  0.589744
'''