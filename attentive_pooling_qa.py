import tensorflow as tf 
import numpy as np
class QA_CNN_Attentive(object):
	def __init__(self,max_input_left,max_input_right,batch_size,vocab_size,embeddings,embedding_size,
		num_filters,dropout_keep_prob,embedding = None,l2_reg_lambda = 0.0,is_Embedding_Needed = False,trainable = False):

		self.question = tf.placeholder(tf.int32,[None,max_input_left],name = 'input_question')
		self.answer = tf.placeholder(tf.int32,[None,max_input_right],name = 'input_answer')
		self.answer_negative = tf.placeholder(tf.int32,[None,max_input_right],name = 'input_right')

		self.dropout_keep_prob = dropout_keep_prob
		self.num_filters = num_filters
		self.embeddings = embeddings
		self.embedding_size = embedding_size
		self.batch_size = batch_size
		self.l2_reg_lambda = l2_reg_lambda
		with tf.name_scope('embedding'):
			if is_Embedding_Needed:
				print "load embedding"
				W = tf.Variable(np.array(self.embeddings),name="W" ,dtype="float32",trainable = trainable)
			else:
				W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name="W",trainable = trainable)
	        self.embedding_W = W
	        self.embedded_chars_q = tf.expand_dims(tf.nn.embedding_lookup(W,self.question),-1)
	        self.embedded_chars_a = tf.expand_dims(tf.nn.embedding_lookup(W,self.answer),-1)
	        self.embedded_chars_a_neg = tf.expand_dims(tf.nn.embedding_lookup(W,self.answer_negative),-1)
		with tf.name_scope('conv_left'):

			filter_shape = [5,self.embedding_size,1,self.num_filters]
			print 'filter shape %s' % filter_shape
			self.conv_W = tf.Variable(tf.truncated_normal(filter_shape,stddev = 0.01),name = 'W')
			self.conv_b = tf.Variable(tf.constant(0.0001, shape=[self.num_filters]), name="b")
		with tf.name_scope('attention'):	
			self.U = tf.Variable(tf.truncated_normal(shape = [self.batch_size,self.num_filters,self.num_filters],stddev = 0.01,name = 'U'))
		with tf.name_scope('score'):
			self.score12 = self.attentive_pooling(self.embedded_chars_q,self.embedded_chars_a)  
	    	self.score13 = self.attentive_pooling(self.embedded_chars_q,self.embedded_chars_a_neg)
		with tf.name_scope('loss'):
			self.l2_loss=0
			for para in [self.embedding_W,self.conv_W,self.conv_b,self.U]:
				self.l2_loss+= tf.nn.l2_loss(para)

			self.losses = tf.maximum(0.0, tf.sub(0.05, tf.sub(self.score12, self.score13))) 
			self.loss = tf.reduce_mean(self.losses+self.l2_loss *self.l2_reg_lambda)

			# self.correct = tf.greater(self.score12, self.score13)
			# self.accuracy = tf.reduce_mean(tf.cast(self.correct, "float"), name="accuracy")

			self.correct = tf.equal(0.0, self.losses)
			self.accuracy = tf.reduce_mean(tf.cast(self.correct, "float"), name="accuracy")
	def attentive_pooling(self,input_left,input_right):
		#concolution	
		conv = tf.nn.conv2d(input_left,
			self.conv_W,
			strides = [1, 1, 1, 1],
			padding = "VALID",
			name = "conv")

		# Apply nonlinearity
		Q = tf.reshape(tf.nn.relu(tf.nn.bias_add(conv, self.conv_b),name = 'relu'),[self.batch_size,-1,self.num_filters])

	
		conv = tf.nn.conv2d(input_right,
			self.conv_W,
			strides = [1, 1, 1, 1],
			padding = "VALID",
			name = "conv")

		# Apply nonlinearity
		A = tf.reshape(tf.nn.tanh(tf.nn.bias_add(conv, self.conv_b),name = 'relu'),[self.batch_size,-1,self.num_filters])
		
		G = tf.tanh(tf.matmul(tf.matmul(Q,self.U),\
			A,transpose_b = True),name = 'G')
		# column-wise pooling ,row-wise pooling
		row_pooling = tf.reduce_max(G,1,True,name = 'row_pooling')
		col_pooling = tf.reduce_max(G,2,True,name = 'col_pooling')

		attention_q = tf.nn.softmax(col_pooling,1,name = 'attention_q')
		attention_a = tf.nn.softmax(row_pooling,name = 'attention_a')

		R_q = tf.reshape(tf.matmul(Q,attention_q,transpose_a = 1),[self.batch_size,self.num_filters,-1])
		R_a = tf.reshape(tf.matmul(attention_a,A),[self.batch_size,self.num_filters,-1])
		self.attention_q =row_pooling
		self.attention_a =col_pooling
		norm_1 = tf.sqrt(tf.reduce_sum(tf.mul(R_q,R_q),1))
		norm_2 = tf.sqrt(tf.reduce_sum(tf.mul(R_a,R_a),1))
		score = tf.div(tf.reduce_sum(tf.mul(R_q,R_a),1),tf.mul(norm_1,norm_2))

		return score

def main():
	batch_size = 1
	cnn = QA_CNN(max_input_left = 10,
    	max_input_right = 10,
        batch_size = batch_size,
        vocab_size = 5000,
        embedding_size = 100,
        num_filters = 7, 
        dropout_keep_prob = 1.0,
        l2_reg_lambda=0.0,
        is_embedding_needded = False)

	input_x_1 = np.reshape(np.arange(batch_size * 10),[batch_size,10])
	input_x_2 = np.reshape(np.arange(batch_size * 10),[batch_size,10])
	input_x_3 = np.reshape(np.arange(30,batch_size * 10 + 30),[batch_size,10])
	print input_x_1
	print input_x_2
	print input_x_3
	with tf.Session() as sess:
	    sess.run(tf.global_variables_initializer())
	    feed_dict = {
	    	cnn.question:input_x_1,
	    	cnn.answer:input_x_2,
	        cnn.answer_negative:input_x_3
	    }
	    q,a,a_neg,score13,score12,loss,accuracy = sess.run([cnn.embedded_chars_q,
	    	cnn.embedded_chars_a,cnn.embedded_chars_a_neg,cnn.score13,cnn.score12,cnn.loss,cnn.accuracy],feed_dict)
	    print score12
	    print score13
	    # print a_neg
        
if __name__ == '__main__':
	main()