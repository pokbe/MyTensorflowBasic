import tensorflow as tf

a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b

sess = tf.InteractiveSession()
# 可以直接使用'c.eval()' 而不需要 'sess'
print(c.eval())
sess.close()

with tf.Session():
# 借助with同样可以使用 'c.eval()'
	print(c.eval())
