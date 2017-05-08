import tensorflow as tf
import numpy as np
class QA_CNN_extend(object):
    def __init__(self,max_input_left,max_input_right,batch_size,vocab_size,embedding_size,filter_sizes,
        num_filters,dropout_keep_prob,embeddings = None,l2_reg_lambda = 0.0,is_Embedding_Needed = False,trainable = True,extend_feature_dim = 10):

       

        self.dropout_keep_prob = dropout_keep_prob
        self.num_filters = num_filters
        self.embeddings = embeddings
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.filter_sizes = filter_sizes
        self.l2_reg_lambda = l2_reg_lambda
        self.para = []
        self.extend_feature_dim = extend_feature_dim
        self.max_input_left = max_input_left
        self.max_input_right = max_input_right
        


        self.question = tf.placeholder(tf.int32,[None,max_input_left],name = 'input_question')
        self.answer = tf.placeholder(tf.int32,[None,max_input_right],name = 'input_answer')
        self.answer_negative = tf.placeholder(tf.int32,[None,max_input_right],name = 'input_right')
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
                filter_shape = [filter_size,embedding_size + extend_feature_dim,1,num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev = 0.1), name="W")
                b = tf.Variable(tf.constant(0.0, shape=[num_filters]), name="b")
                self.kernels.append((W,b))
                self.para.append(W)
                self.para.append(b)

        with tf.name_scope('score'):
            q_pos_embedding = self.getEmbedding(self.question,self.q_pos_overlap)
            q_neg_embedding = self.getEmbedding(self.question,self.q_neg_overlap)
            a_pos_embedding = self.getEmbedding(self.answer, self.a_pos_overlap)
            a_neg_embedding = self.getEmbedding(self.answer_negative,self.a_neg_overlap)
            # q_pos_embedding = 0
            # q_pos_embedding = 0
            # a_pos_embedding = 0
            # a_neg_embedding = 0
            self.score12 = self.qa_extend(q_pos_embedding,a_pos_embedding)
            self.score13 = self.qa_extend(q_neg_embedding,a_neg_embedding)
        l2_loss = tf.constant(0.0)
        for p in self.para:
            l2_loss += tf.nn.l2_loss(p)
        with tf.name_scope("loss"):
            self.losses = tf.maximum(0.0, tf.sub(0.05, tf.sub(self.score12, self.score13)))
            self.loss = tf.reduce_sum(self.losses) + l2_reg_lambda * l2_loss
            
        # Accuracy
        with tf.name_scope("accuracy"):
            self.correct = tf.equal(0.0, self.losses)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, "float"), name="accuracy")
    def getEmbedding(self,words_indice,overlap_indice):
        embedded_chars_q = tf.nn.embedding_lookup(self.embedding_W,words_indice)
        overlap_embedding_q = tf.nn.embedding_lookup(self.overlap_W,overlap_indice)
        return  tf.expand_dims(tf.concat(2,[embedded_chars_q,overlap_embedding_q]),-1)
    def qa_extend(self,q_concat_embedding,a_concat_embedding):

        pooled_outputs_1 = []
        pooled_outputs_2 = []

        for i,filter_size in enumerate(self.filter_sizes):
            conv = tf.nn.conv2d(
                    q_concat_embedding,
                    self.kernels[i][0],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="conv-1"
            )

            h = tf.nn.relu(tf.nn.bias_add(conv, self.kernels[i][1]), name="relu-1")

            pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.max_input_left - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="poll-1"
            )
            pooled_outputs_1.append(pooled)

            conv = tf.nn.conv2d(
                    a_concat_embedding,
                    self.kernels[i][0],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="conv-2"
            )
            print conv
            h = tf.nn.relu(tf.nn.bias_add(conv, self.kernels[i][1]), name="relu-1")

            pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.max_input_right - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="poll-2"
            )
            pooled_outputs_2.append(pooled)
        num_filters_total = self.num_filters * len(self.filter_sizes)
        pooled_reshape_1 = tf.reshape(tf.concat(3, pooled_outputs_1), [-1, num_filters_total]) 
        pooled_reshape_2 = tf.reshape(tf.concat(3, pooled_outputs_2), [-1, num_filters_total]) 

        pooled_flat_1 = tf.nn.dropout(pooled_reshape_1, self.dropout_keep_prob)
        pooled_flat_2 = tf.nn.dropout(pooled_reshape_2, self.dropout_keep_prob)
        
        pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.mul(pooled_flat_1, pooled_flat_1), 1)) 
        pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.mul(pooled_flat_2, pooled_flat_2), 1))
        pooled_mul_12 = tf.reduce_sum(tf.mul(pooled_flat_1, pooled_flat_2), 1) 
        score = tf.div(pooled_mul_12, tf.mul(pooled_len_1, pooled_len_2), name="scores") 
        return score
    
if __name__ == '__main__':
    cnn = QA_CNN_extend(max_input_left = 33,
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
    input_x_1 = np.reshape(np.arange(3 * 33),[3,33])
    input_x_2 = np.reshape(np.arange(3 * 40),[3,40])
    input_x_3 = np.reshape(np.arange(3 * 40),[3,40])

    q_pos_embedding = np.reshape(np.arange(3 * 33 * 10),[3,33,10])
    q_neg_embedding = np.reshape(np.arange(3 * 33 * 10),[3,33,10])
    a_pos_embedding = np.reshape(np.arange(3 * 40 * 10),[3,40,10])
    a_neg_embedding = np.reshape(np.arange(3 * 40 * 10),[3,40,10])    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = {
            cnn.question:input_x_1,
            cnn.answer:input_x_2,
            cnn.answer_negative:input_x_3,
            cnn.q_pos_feature_embedding:q_pos_embedding,
            cnn.q_neg_feature_embedding:q_neg_embedding,
            cnn.a_pos_feature_embedding:a_pos_embedding,
            cnn.a_neg_feature_embedding:a_neg_embedding

        }
        question,answer,score = sess.run([cnn.question,cnn.answer,cnn.score12],feed_dict)
        print question.shape,answer.shape
        print score 

