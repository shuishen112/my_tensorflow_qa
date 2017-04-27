import tensorflow as tf


scores= tf.reshape (tf.Variable(tf.random_uniform([100,1], -1.0, 1.0)),[100])
input_y=tf.reshape ( tf.Variable(tf.zeros([100,1])),[100])
t1 = tf.ones((3,2),dtype = 'int32')
t2 = [7,8,9,5,6,7,3,3,3]
t2 = tf.reshape(t2,(3,3))
result=tf.nn.softmax_cross_entropy_with_logits(logits=scores+0.00001, labels=input_y+0.00001)
with  tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print t1.eval()
	print sess.run(scores)