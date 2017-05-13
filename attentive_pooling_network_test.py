import tensorflow as tf
import numpy as np
class QA_attentive(object):
    def __init__(self,max_input_left,max_input_right,batch_size,vocab_size,embedding_size,filter_sizes,
        num_filters,dropout_keep_prob,embeddings = None,l2_reg_lambda = 0.0,learning_rate = 0.001,is_Embedding_Needed = False,trainable = True,extend_feature_dim = 10,model_type = 'qacnn'):

        self.question = tf.placeholder(tf.int32,[None,max_input_left],name = 'input_question')
        self.answer = tf.placeholder(tf.int32,[None,max_input_right],name = 'input_answer')
        self.answer_negative = tf.placeholder(tf.int32,[None,max_input_right],name = 'input_right')
        self.dropout_keep_prob = dropout_keep_prob
        self.num_filters = num_filters
        self.max_input_left = max_input_left
        self.max_input_right = max_input_right
        self.embeddings = embeddings
        self.extend_feature_dim = extend_feature_dim
        self.total_embedding_size = embedding_size + extend_feature_dim 
        self.filter_sizes = filter_sizes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.l2_reg_lambda = l2_reg_lambda
        self.para = []
        self.q_pos_overlap = tf.placeholder(tf.int32,[None,max_input_left],name = 'q_pos_feature_embed')
        self.q_neg_overlap = tf.placeholder(tf.int32,[None,max_input_left],name = 'q_neg_feature_embed')
        self.a_pos_overlap = tf.placeholder(tf.int32,[None,max_input_right],name = 'a_feature_embed')
        self.a_neg_overlap = tf.placeholder(tf.int32,[None,max_input_right],name = 'a_neg_feature_embed')
        with tf.name_scope('embedding'):
            if is_Embedding_Needed:
                print "load embedding"
                W = tf.Variable(np.array(self.embeddings),name="W" ,dtype="float32",trainable = trainable)
            else:
                W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name="W",trainable = trainable)
            self.embedding_W = W
            self.overlap_W = tf.Variable(tf.random_uniform([3, self.extend_feature_dim], -1.0, 1.0),name="W",trainable = trainable)
            self.para.append(self.embedding_W)
            self.para.append(self.overlap_W)
        self.kernels = []
        for i,filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-max-pool-%s' % filter_size):
                filter_shape = [filter_size,self.total_embedding_size,1,num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev = 0.1), name="W")
                b = tf.Variable(tf.constant(0.0, shape=[num_filters]), name="b")
                self.kernels.append((W,b))
                self.para.append(W)
                self.para.append(b)
        with tf.name_scope('concat_embeddings'):
            q_pos_embedding = self.getEmbedding(self.question,self.q_pos_overlap)
            q_neg_embedding = self.getEmbedding(self.question,self.q_neg_overlap)
            a_pos_embedding = self.getEmbedding(self.answer, self.a_pos_overlap)
            a_neg_embedding = self.getEmbedding(self.answer_negative,self.a_neg_overlap)
        with tf.name_scope('feature_map'):
            q_pos_f = self.get_feature_map(q_pos_embedding)
            q_neg_f = self.get_feature_map(q_neg_embedding)
            a_pos_f = self.get_feature_map(a_pos_embedding)
            a_neg_f = self.get_feature_map(a_neg_embedding)
        with tf.name_scope('attention'):    
            self.U = tf.Variable(tf.truncated_normal(shape = [self.num_filters * len(filter_sizes),\
                self.num_filters * len(filter_sizes)],stddev = 0.01,name = 'U'))
            self.para.append(self.U)
        with tf.name_scope('score'):
            self.score12 = self.attentive_pooling(q_pos_f,a_pos_f)
            self.score13 = self.attentive_pooling(q_neg_f,a_neg_f)
        #     self.embedded_chars_q = tf.expand_dims(tf.nn.embedding_lookup(W,self.question),-1)
        #     self.embedded_chars_a = tf.expand_dims(tf.nn.embedding_lookup(W,self.answer),-1)
        #     self.embedded_chars_a_neg = tf.expand_dims(tf.nn.embedding_lookup(W,self.answer_negative),-1)
    
        '''
        pooled_outputs_1 = []
        pooled_outputs_2 = []
        pooled_outputs_3 = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-%s" % filter_size):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev = 0.1), name="W")
                b = tf.Variable(tf.constant(0.0, shape=[num_filters]), name="b")
                self.conv_W = W
                self.conv_b = b
                self.para.append(W)
                self.para.append(b)
                conv = tf.nn.conv2d(
                    self.embedded_chars_q,
                    W,
                    strides=[1, 1, embedding_size, 1],
                    padding='SAME',
                    name="conv-1"
                )

                print conv
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu-1")

                pooled_outputs_1.append(h)

                conv = tf.nn.conv2d(
                    self.embedded_chars_a,
                    W,
                    strides=[1, 1, embedding_size, 1],
                    padding='SAME',
                    name="conv-2"
                )

                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu-2")
                pooled_outputs_2.append(h)

                conv = tf.nn.conv2d(
                    self.embedded_chars_a_neg,
                    W,
                    strides=[1, 1, embedding_size, 1],
                    padding='SAME',
                    name="conv-3"
                )

                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu-3")
                pooled_outputs_3.append(h)


        self.pooled_flat_1 = tf.concat(3, pooled_outputs_1)
        self.pooled_flat_2 = tf.concat(3, pooled_outputs_2)
        self.pooled_flat_3 = tf.concat(3, pooled_outputs_3)
        '''

        # with tf.name_scope('score'):
        #     self.score12 = self.attentive_pooling(self.pooled_flat_1,self.pooled_flat_2)
        #     self.score13 = self.attentive_pooling(self.pooled_flat_1,self.pooled_flat_3)

        with tf.name_scope('loss'):
            self.l2_loss=0
            for para in self.para:
                self.l2_loss+= tf.nn.l2_loss(para)

            self.losses = tf.maximum(0.0, tf.sub(0.05, tf.sub(self.score12, self.score13))) 
            self.loss = tf.reduce_mean(self.losses+self.l2_loss *self.l2_reg_lambda)

            # self.correct = tf.greater(self.score12, self.score13)
            # self.accuracy = tf.reduce_mean(tf.cast(self.correct, "float"), name="accuracy")

            self.correct = tf.equal(0.0, self.losses)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, "float"), name="accuracy")  
    def get_feature_map(self,sentence_indice):
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-%s" % filter_size):
               
                conv = tf.nn.conv2d(
                    sentence_indice,
                    self.kernels[i][0],
                    strides=[1, 1, self.total_embedding_size, 1],
                    padding='SAME',
                    name="conv-1"
                )
                print conv
                h = tf.nn.relu(tf.nn.bias_add(conv, self.kernels[i][1]), name="relu-1")

                pooled_outputs.append(h)
        self.pooled_flat = tf.concat(3, pooled_outputs)
        return self.pooled_flat
    def getEmbedding(self,words_indice,overlap_indice):
        embedded_chars_q = tf.nn.embedding_lookup(self.embedding_W,words_indice)
        overlap_embedding_q = tf.nn.embedding_lookup(self.overlap_W,overlap_indice)
        return  tf.expand_dims(tf.concat(2,[embedded_chars_q,overlap_embedding_q]),-1)
            # self.score13 = self.attentive_pooling(self.pooled_flat_1,self.pooled_flat_3)
    def attentive_pooling(self,input_left,input_right):
        Q = tf.reshape(input_left,[self.batch_size,self.max_input_left,len(self.filter_sizes) * self.num_filters],name = 'Q')
        A = tf.reshape(input_right,[self.batch_size,self.max_input_right,len(self.filter_sizes) * self.num_filters],name = 'A')

        G = tf.tanh(tf.matmul(tf.matmul(Q,self.U),\
        A,transpose_b = True),name = 'G')

        # column-wise pooling ,row-wise pooling
        row_pooling = tf.reduce_max(G,1,True,name = 'row_pooling')
        col_pooling = tf.reduce_max(G,2,True,name = 'col_pooling')

        attention_q = tf.nn.softmax(col_pooling,1,name = 'attention_q')
        attention_a = tf.nn.softmax(row_pooling,name = 'attention_a')

        R_q = tf.reshape(tf.matmul(Q,attention_q,transpose_a = 1),[self.batch_size,self.num_filters * len(self.filter_sizes),-1],name = 'R_q')
        R_a = tf.reshape(tf.matmul(attention_a,A),[self.batch_size,self.num_filters * len(self.filter_sizes),-1],name = 'R_a')

        norm_1 = tf.sqrt(tf.reduce_sum(tf.mul(R_q,R_q),1))
        norm_2 = tf.sqrt(tf.reduce_sum(tf.mul(R_a,R_a),1))
        score = tf.div(tf.reduce_sum(tf.mul(R_q,R_a),1),tf.mul(norm_1,norm_2))
        print score

        return score


        # print pooled_outputs_1
        # print pooled_outputs_2
        # print pooled_outputs_3
if __name__ == '__main__':
     cnn = QA_attentive(max_input_left = 33,
        max_input_right = 40,
        batch_size = 3,
        vocab_size = 5000,
        embedding_size = 100,
        filter_sizes = [3,4,5],
        num_filters = 64, 
        dropout_keep_prob = 1.0,
        embeddings = None,
        l2_reg_lambda=0.0,
        is_Embedding_Needed = False,
        trainable = True,
        extend_feature_dim = 10)