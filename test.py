# -*- coding:utf-8-*-
'''
import tensorflow as tf

#test convolution
x = tf.constant([[1., 2., 3.],
                 [4., 5., 6.]])

x = tf.reshape(x, [1, 2, 3, 1])  # give a shape accepted by tf.nn.max_pool

valid_pad = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
same_pad = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

print valid_pad.get_shape() == [1, 1, 1, 1]  # valid_pad is [5.]
print same_pad.get_shape() == [1, 1, 2, 1]   # same_pad is  [5., 6.]

with tf.Session() as sess:
	print x.get_shape()
	'''
# import numpy as np
# words = dict()
# text = ['who','are','you']
# for t in text:
# 	words[t] = 0
	
# file = open('data/trec/train.txt')
# for e,line in enumerate(file):
# 	# if e < 5:
# 		for token in line.strip().split():
# 			if token in words:
# 				words[token] += 1	
# 	# else:
# 		# break
# print words
import tensorflow as tf
import numpy as np
import numpy
def testMul():
	# batch mul
	batch = 3
	input_left = tf.placeholder(tf.float32, shape=[batch,4,7], name='input_left')
	input_right = tf.placeholder(tf.float32,shape = [batch,4,7],name = 'inpout_right')

	q = np.arange(batch * 4 * 7).reshape((batch,4,7))
	a = np.arange(batch * 4 * 7).reshape(batch,4,7)

	G = tf.matmul(input_left,input_right,transpose_b = True)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		sess.run(input_left,feed_dict = {input_left:q,input_right:a})
		print input_left
		print input_right
		print G
def testReshape():
	a = tf.Variable(tf.random_uniform([200000]))
	b = tf.reshape(a,[-1,100,100])
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print b.eval()
def testDropout():
	a = tf.Variable(tf.random_uniform([10,10],-1,1),name = 'a')
	a_drop = tf.nn.dropout(a,keep_prob = 1)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print sess.run(a_drop)
def pooling():
	batch = 3
	input_left = tf.placeholder(tf.float32, shape=[batch,4,5], name='input_left')
	input_right = tf.placeholder(tf.float32,shape = [batch,4,7],name = 'inpout_right')

	U = tf.Variable(tf.truncated_normal(shape = [batch,4,4],stddev = 0.1,name = 'U'))
	# pooling = tf.reduce_max(x,1)
	# raw_pooling= tf.reshape( tf.reduce_max(x,1,True),[batch,-1])
	# row_pooling= tf.reshape( tf.reduce_max(x,1,True),[batch,-1])

	# input_right_transpose = tf.transpose(input_right,perm = [0,2,1])

	G = tf.tanh(tf.matmul(tf.matmul(input_right,U,transpose_a = True),\
		input_left),name = 'G')

	# column-wise pooling ,row-wise pooling
	row_pooling = tf.reduce_max(G,1,True,name = 'row_pooling')
	col_pooling = tf.reduce_max(G,2,True,name = 'col_pooling')

	attention_q = tf.nn.softmax(row_pooling,name = 'attention_q')
	attention_a = tf.nn.softmax(col_pooling,name = 'attention_a')

	R_q = tf.matmul(input_left,attention_q,transpose_b = True)
	R_a = tf.matmul(input_right,attention_a)
	q = np.arange(batch * 4 * 5).reshape((batch,4,5)) 
	a = np.arange(batch * 4 * 7).reshape(batch,4,7) 
	
	norm_1 = tf.sqrt(tf.reduce_sum(tf.mul(R_q,R_q),1))
	norm_2 = tf.sqrt(tf.reduce_sum(tf.mul(R_a,R_a),1))

	# norm_q = tf.div(R_q,norm_1)
	# norm_a = tf.div(R_a,norm_2)
	# norm_a = tf.div(R_a,norm_2)
	score = tf.div(tf.reduce_sum(tf.mul(R_q,R_a),1),tf.mul(norm_1,norm_2))
	# score = tf.reduce_sum(tf.mul(norm_q,norm_a),1)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		# print(sess.run(raw_pooling,feed_dict={x:data}))
		# print(sess.run(row_pooling,feed_dict={x:data}))
		# print(sess.run(pooling,feed_dict = {x:data}))
		score = sess.run([score],feed_dict = {input_left:q,input_right:a})
		# R_q,R_a,norm_1,norm_2,norm_q,norm_a =  sess.run([R_q,R_a,norm_1,norm_2,norm_q,norm_a],feed_dict = {input_left:q,input_right:a})
		print norm_1
		print norm_2
		print score
		# print norm_1
		# print norm_a
		# print score
		# print norm_a
		# print a
		# print G
		# print attention_q
		# print attention_a
		# print R_q
		# print R_a

def testConcolution():
	batch_size = 3
	filter_size = 2 #n-gram
	embedding_size = 50
	num_filter = 5
	q = np.reshape(np.arange(batch_size * 7),(batch_size,7))
	a = np.reshape(np.arange(20,batch_size * 7+20),(batch_size,7))
	print q
	print a
	input_left = tf.placeholder(tf.int32,shape = [batch_size,7],name = 'question')
	input_right = tf.placeholder(tf.int32,shape = [batch_size,7],name = 'answer')
	# embedding
	# input_right = tf.placeholder(tf.int32,shape = [3,5,5])
	with tf.name_scope('embedding'):
		W = tf.Variable(tf.random_uniform([100,embedding_size]))
		embedding_char_left = tf.expand_dims(tf.nn.embedding_lookup(W,input_left),-1)
		embedding_char_right = tf.expand_dims(tf.nn.embedding_lookup(W,input_right),-1)

	
	#concolution
	filter_shape = [filter_size,embedding_size,1,num_filter]
	print 'filter shape %s' % filter_shape
	with tf.name_scope('conv_left'):
		# W = tf.Variable(tf.truncated_normal(filter_shape,stddev = 0.1),name = 'W')
		W = tf.Variable(tf.ones(filter_shape),name = 'W')
		b = tf.Variable(tf.constant(0.1, shape=[num_filter]), name="b")
		conv = tf.nn.conv2d(embedding_char_left,
			W,
			strides = [1, 1, 1, 1],
			padding = "VALID",
			name = "conv")

		# Apply nonlinearity
		Q = tf.reshape(tf.nn.relu(tf.nn.bias_add(conv, b),name = 'relu'),[batch_size,-1,num_filter])

		#column-wise max-pooling
		# col_pooling = tf.reduce_max(Q,1,True)
	with tf.name_scope('conv_right'):
		# W = tf.Variable(tf.truncated_normal(filter_shape,stddev = 0.1),name = 'W')
		W = tf.Variable(tf.ones(filter_shape),name = 'W')
		b = tf.Variable(tf.constant(0.1, shape=[num_filter]), name="b")
		conv = tf.nn.conv2d(embedding_char_right,
			W,
			strides = [1, 1, 1, 1],
			padding = "VALID",
			name = "conv")

		# Apply nonlinearity
		A = tf.reshape(tf.nn.relu(tf.nn.bias_add(conv, b),name = 'relu'),[batch_size,-1,num_filter])
	with tf.name_scope('attention'):	
		# U = tf.Variable(tf.truncated_normal(shape = [batch_size,num_filter,num_filter],stddev = 0.1,name = 'U'))
		U = tf.Variable(tf.ones(shape = [batch_size,num_filter,num_filter]),name = 'U')
		G = tf.matmul(tf.matmul(Q,U),\
			A,transpose_b = True)
		# column-wise pooling ,row-wise pooling
		row_pooling = tf.reduce_max(G,1,True,name = 'row_pooling')
		col_pooling = tf.reduce_max(G,2,True,name = 'col_pooling')

		attention_q = tf.nn.softmax(col_pooling,name = 'attention_q')
		attention_a = tf.nn.softmax(row_pooling,name = 'attention_a')
		R_q = tf.reshape(tf.matmul(Q,attention_q,transpose_a = 1),[batch_size,num_filter,-1])
		R_a = tf.reshape(tf.matmul(attention_a,A),[batch_size,num_filter,-1])
	with tf.name_scope('score'):
		norm_1 = tf.sqrt(tf.reduce_sum(tf.mul(R_q,R_q),1))
		norm_2 = tf.sqrt(tf.reduce_sum(tf.mul(R_a,R_a),1))

		score = tf.div(tf.reduce_sum(tf.mul(R_q,R_a),1),tf.mul(norm_1,norm_2))
	with tf.Session() as sess:
		writer = tf.summary.FileWriter('./graphs',sess.graph)
		sess.run(tf.global_variables_initializer())
		# print sess.run(W)
		G,q,a,score = sess.run([G,attention_q,attention_a,score],feed_dict = {input_left:q,input_right:a})
		# print embedding[0][0][1] == sess.run(W[1])
		# print sess.run(W[1])
		writer.close()
		# print G
		# print Q
		# print A
		# print attention_q
		# print attention_a
		# print q
		# print a
		# print norm_1
		# print norm_2
		print q.shape,a.shape
		print q
		print a
		# print G
		print score
def test_ones():
	a = tf.ones([100,100,100])
	with tf.Session() as sess:
		print a.eval()
def testU():
	batch_size = 3
	num_filter = 5
	Q = tf.Variable(tf.random_uniform(shape = (batch_size,3,num_filter)))
	U = tf.Variable(tf.ones(shape = (num_filter,num_filter)))
	A = tf.Variable(tf.random_uniform(shape = (batch_size,4,num_filter)))
	first = tf.matmul(tf.reshape(Q,[-1,num_filter]),U)
	second_step = tf.reshape(first,[batch_size,-1,num_filter])
	result = tf.batch_matmul(second_step,tf.transpose(A,perm = [0,2,1]))
	sum_ = tf.reduce_sum(Q,2)
	print result
	print second_step
	print first
	print U
	print first
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		Q,first,U,sum_ = sess.run([Q,first,U,sum_])
		print Q
		print U
		print first
		print sum_
def test_log():
	y = tf.constant([0.0,1.0])
	p = tf.Variable(tf.random_uniform(shape = [1,2]))
	loss1 = tf.reduce_sum(y * tf.log(p),reduction_indices = [1])
	loss2 = tf.nn.softmax_cross_entropy_with_logits(y,p)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		loss1,loss2,p = sess.run([loss1,loss2,p])
		print np.log(p)
		print loss2
		print loss1
def test_similarity():
	num_filters_total = 10
	left = tf.Variable(tf.random_uniform(shape = [3,10]))
	right = tf.Variable(tf.random_uniform(shape = [4,10]))
	W = tf.get_variable(
        "W",
        shape=[num_filters_total, num_filters_total],
        initializer=tf.contrib.layers.xavier_initializer())
	transform = tf.matmul(left,W)
	sims = tf.reduce_sum(tf.mul(transform, right), 1, keep_dims=True)
	print sims
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print sess.run(left)
def seefile():
	file = 'missword'
	import pandas as pd
	import pynlpir
	pynlpir.open()
	df = pd.read_csv(file,header = None,names = ['word'])
	number = r'\d+'
	df = df[df['word'].str.contains(number) == False]
	english = r'^[A-Za-z]+$'
	# df[df['word'].str.contains(number) == False].to_csv('unknow')
	word = df[df['word'].str.contains(english) == False]
	for w in word['word']:
		print '#'.join(pynlpir.segment(w,pos_tagging = False))
	# print df
if __name__ == '__main__':
	# test_log()
	# mygenerator = (x*x for x in range(3))
	# for i in mygenerator:
	# 	print i
	# for i in mygenerator:
	# 	print i
	# a = [[1,2,3],[4,5,6],[7,8,9]]
	# import itertools
	# merged = list(itertools.chain(*a))
	# print merged
	testDropout()
	# seefile()
	# def createGenerator():
	# 	mylist = range(3)
	# 	for i in mylist:
	# 		yield i * i,i * i * i
	# for j in range(5):
	# 	mygenerator = createGenerator()
	# 	for i in mygenerator:
	# 		print i
	# a = np.arange(100,200)
	# a = map(str,a)
	# index = np.random.choice(a,size = [30])
	# print index
	# a = ['A partly submerged glacier cave on Perito Moreno Glacier .', 'The ice facade is approximately 60 m high', 'Ice formations in the Titlis glacier cave', 'Glacier caves are often called ice caves , but this term is properly used to describe bedrock caves that contain year-round ice .']

	# b = np.random.choice(a,2)
	# print a[0]
	# import chardet
	# print chardet.detect(a[0])
	# testU()

	# testReshape()
	# test_ones()
	# testConcolution()W
	# testDropout()
	# print float(8) / 17
	# pooling()
	# testMul()