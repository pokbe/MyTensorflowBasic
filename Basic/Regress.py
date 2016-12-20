import numpy as np
import tensorflow as tf

X_data = np.linspace(-10,10,100)
Y_data = X_data * 3.0 + 1.0 + np.random.normal(0.0,2.0,X_data.shape)

def mymodel(X,W,b):
	Y = tf.add(tf.mul(X,W),b)
	return Y

input_X = tf.placeholder("float")
input_Y = tf.placeholder("float")

W = tf.Variable(0.0,dtype=tf.float32)
b = tf.Variable(0.0,dtype=tf.float32)

pre_Y = mymodel(input_X, W, b)

cost = tf.pow((pre_Y-input_Y),2)
op = tf.train.AdamOptimizer().minimize(cost)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for e in range(300):
	for (x,y) in zip(X_data,Y_data):
		sess.run(op,feed_dict={input_X:x, input_Y:y})
	(temp_W , temp_b) = sess.run((W,b))
	print("W: %f , b: %f"%(temp_W,temp_b))

sess.close()