import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

epochs = 500
batch_size = 100

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

input_data = tf.placeholder(tf.float32,[None,784])
input_label = tf.placeholder(tf.float32,[None,10])

W = tf.Variable(tf.random_normal([784,10]))
b = tf.Variable(tf.random_normal([10]))

logit_data = tf.add(tf.matmul(input_data,W),b)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logit_data,input_label))
op = tf.train.GradientDescentOptimizer(0.02).minimize(cost)

predict = tf.nn.softmax(logit_data)
correct = tf.cast(tf.equal(tf.argmax(predict,1),tf.argmax(input_label,1)),tf.float32)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for epoch in range(epochs):
	batch_total = int(mnist.train.num_examples/batch_size)
	cost_sum = 0.0
	for batch in range(batch_total):
		batch_data , batch_label = mnist.train.next_batch(batch_size)
		_ , batch_cost = sess.run([op,cost],feed_dict={input_data:batch_data, input_label:batch_label})
		cost_sum += batch_cost
	cost_avg = cost_sum/batch_total
	print("Epoch ", epoch ,"Cost: ",cost_avg)
print("Training Done !")

cor = sess.run(correct,feed_dict={input_data:mnist.test.images, input_label:mnist.test.labels})
correct_rate = tf.reduce_sum(cor)/mnist.test.num_examples
print("Test Accuracy: " ,cor_rate)
sess.close()
