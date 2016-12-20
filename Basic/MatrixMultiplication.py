import numpy as np
import tensorflow as tf

A = tf.constant([[2.0,2.0]])
B = tf.constant([[5.0,],[5.0,]])

C = tf.matmul(A,B)
D = tf.matmul(B,A)

with tf.Session() as sess:
	print("C: ",sess.run(C))
	print("D: ",sess.run(D))