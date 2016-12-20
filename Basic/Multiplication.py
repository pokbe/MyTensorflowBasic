import tensorflow as tf
import numpy as np

a = tf.placeholder("float")
b = tf.placeholder("float")

c = tf.mul(a,b)

sess = tf.Session()
x = 3
y = 4
print("%f * %f = %f"%(x,y,sess.run(c,feed_dict={a:x,b:y})))