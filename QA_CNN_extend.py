import tensorflow as tf
import numpy as np


# model_type :apn or qacnn
class QA_CNN_extend(object):
    def __init__(self,max_input_left,max_input_right,batch_size,vocab_size,embedding_size,filter_sizes,num_filters,
        dropout_keep_prob=1,learning_rate=0.001,embeddings = None,l2_reg_lambda = 0.0,overlap_needed = False,trainable = True,extend_feature_dim = 10,model_type="qacnn"):

       

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
        self.overlap_needed = overlap_needed
        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        if self.overlap_needed:
            self.total_embedding_dim = embedding_size + extend_feature_dim
        else:
            self.total_embedding_dim = embedding_size
        self.learning_rate=learning_rate

        self.question = tf.placeholder(tf.int32,[None,max_input_left],name = 'input_question')
        self.answer = tf.placeholder(tf.int32,[None,max_input_right],name = 'input_answer')
        self.answer_negative = tf.placeholder(tf.int32,[None,max_input_right],name = 'input_right')
        self.q_pos_overlap = tf.placeholder(tf.int32,[None,max_input_left],name = 'q_pos_feature_embed')
        self.q_neg_overlap = tf.placeholder(tf.int32,[None,max_input_left],name = 'q_neg_feature_embed')
        self.a_pos_overlap = tf.placeholder(tf.int32,[None,max_input_right],name = 'a_feature_embed')
        self.a_neg_overlap = tf.placeholder(tf.int32,[None,max_input_right],name = 'a_neg_feature_embed')
        with tf.name_scope('embedding'):
            if embeddings != None:
                print "load embedding"
                W = tf.Variable(np.array(self.embeddings),name="W" ,dtype="float32",trainable = trainable)
                
            else:
                print "random embedding"
                W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name="W",trainable = trainable)
            self.embedding_W = W
            # a = np.zeros((3,self.extend_feature_dim),dtype = 'float32')
            # a[1,:] = 1
            # a[2,:] = 2
            self.overlap_W = tf.Variable(tf.random_uniform([3, self.extend_feature_dim], -1.0, 1.0),name="W",trainable = True)
            # self.overlap_W = tf.Variable(a,name="W",trainable = True)
            self.para.append(self.embedding_W)
            self.para.append(self.overlap_W)
        self.kernels = []
        for i,filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-max-pool-%s' % filter_size):
                filter_shape = [filter_size,self.total_embedding_dim,1,num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev = 0.1), name="W")
                b = tf.Variable(tf.constant(0.0, shape=[num_filters]), name="b")
                self.kernels.append((W,b))
                self.para.append(W)
                self.para.append(b)

        if model_type=="apn":
            with tf.name_scope('attention'):    
                self.U = tf.Variable(tf.truncated_normal(shape = [self.num_filters * len(filter_sizes),\
                    self.num_filters * len(filter_sizes)],stddev = 0.01,name = 'U'))
                self.para.append(self.U)

        with tf.name_scope('score'):
            q_pos_embedding = self.getEmbedding(self.question,self.q_pos_overlap)
            q_neg_embedding = self.getEmbedding(self.question,self.q_neg_overlap)
            a_pos_embedding = self.getEmbedding(self.answer, self.a_pos_overlap)
            a_neg_embedding = self.getEmbedding(self.answer_negative,self.a_neg_overlap)
            embeddings=[q_pos_embedding,q_neg_embedding,a_pos_embedding,a_neg_embedding]
            if model_type=="qacnn":

                q_pos_feature_map,q_neg_feature_map,a_pos_feature_map,a_neg_feature_map= [self.getFeatureMap(embedding,right=i/2) for i,embedding in enumerate(embeddings) ]

                self.score12 = self.getCosine(q_pos_feature_map,a_pos_feature_map)
                self.score13 = self.getCosine(q_neg_feature_map,a_neg_feature_map)
            elif model_type=="apn":

                q_pos_feature_map,q_neg_feature_map,a_pos_feature_map,a_neg_feature_map= [self.getFeatureMapWithPooling(embedding,right=i/2) for i,embedding in enumerate(embeddings) ]
                self.score12 = self.attentive_pooling(q_pos_feature_map,a_pos_feature_map)
                self.score13 = self.attentive_pooling(q_neg_feature_map,a_neg_feature_map)
            else:
                print "no implement"
                exit(0)
            
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
        self.global_step = tf.Variable(0, name="global_step", trainable = False)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step = self.global_step)
        
    def getEmbedding(self,words_indice,overlap_indice):
        embedded_chars_q = tf.nn.embedding_lookup(self.embedding_W,words_indice)
        if not self.overlap_needed:
            return  tf.expand_dims(embedded_chars_q,-1)
        overlap_embedding_q = tf.nn.embedding_lookup(self.overlap_W,overlap_indice)
        return  tf.expand_dims(tf.concat(2,[embedded_chars_q,overlap_embedding_q]),-1)

    def getFeatureMap(self,embedding,right=True):
        if right==1:
            max_length=self.max_input_right
        else:
            max_length=self.max_input_left
        pooled_outputs = []       
        for i,filter_size in enumerate(self.filter_sizes):
            conv = tf.nn.conv2d(
                    embedding,
                    self.kernels[i][0],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="conv-1"
            )
            h = tf.nn.relu(tf.nn.bias_add(conv, self.kernels[i][1]), name="relu-1")

            pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, max_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="poll-1"
            )
            pooled_outputs.append(pooled) 
        pooled_reshape = tf.reshape(tf.concat(3, pooled_outputs), [-1, self.num_filters_total])  
        return pooled_reshape

    def getFeatureMapWithPooling(self,embedding,right=True):
        cnn_outputs = []       
        for i,filter_size in enumerate(self.filter_sizes):
            

            conv = tf.nn.conv2d(
                    embedding,
                    self.kernels[i][0],
                    strides=[1, 1, self.total_embedding_dim, 1],
                    padding='SAME',
                    name="conv-1"
                )

            
            h = tf.nn.relu(tf.nn.bias_add(conv, self.kernels[i][1]), name="relu-1")
            cnn_outputs.append(h)
            
            # cnn_reshaped = tf.reshape(tf.concat(3, cnn_outputs), [-1, self.num_filters_total]) 
        cnn_reshaped = tf.concat(3, cnn_outputs)
        return cnn_reshaped

    def getCosine(self,q,a):
        pooled_flat_1 = tf.nn.dropout(q, self.dropout_keep_prob)
        pooled_flat_2 = tf.nn.dropout(a, self.dropout_keep_prob)
        
        pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.mul(pooled_flat_1, pooled_flat_1), 1)) 
        pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.mul(pooled_flat_2, pooled_flat_2), 1))
        pooled_mul_12 = tf.reduce_sum(tf.mul(pooled_flat_1, pooled_flat_2), 1) 
        score = tf.div(pooled_mul_12, tf.mul(pooled_len_1, pooled_len_2), name="scores") 
        return score    
        
    def attentive_pooling(self,input_left,input_right):
        Q = tf.reshape(input_left,[self.batch_size,self.max_input_left,len(self.filter_sizes) * self.num_filters],name = 'Q')
        A = tf.reshape(input_right,[self.batch_size,self.max_input_right,len(self.filter_sizes) * self.num_filters],name = 'A')

        # G = tf.tanh(tf.matmul(tf.matmul(Q,self.U),\
        # A,transpose_b = True),name = 'G')
        first = tf.matmul(tf.reshape(Q,[-1,len(self.filter_sizes) * self.num_filters]),self.U)
        second_step = tf.reshape(first,[self.batch_size,-1,len(self.filter_sizes) * self.num_filters])
        result = tf.batch_matmul(second_step,tf.transpose(A,perm = [0,2,1]))
        G = tf.tanh(result)
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
        overlap_needed = True,
        trainable = True,
        extend_feature_dim = 10)
    input_x_1 = np.reshape(np.arange(3 * 33),[3,33])
    input_x_2 = np.reshape(np.arange(3 * 40),[3,40])
    input_x_3 = np.reshape(np.arange(3 * 40),[3,40])

    q_pos_embedding = np.ones((3,33))
    q_neg_embedding = np.ones((3,33))
    a_pos_embedding = np.ones((3,40))
    a_neg_embedding = np.ones((3,40))    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = {
            cnn.question:input_x_1,
            cnn.answer:input_x_2,
            cnn.answer_negative:input_x_3,
            cnn.q_pos_overlap:q_pos_embedding,
            cnn.q_neg_overlap:q_neg_embedding,
            cnn.a_pos_overlap:a_pos_embedding,
            cnn.a_neg_overlap:a_neg_embedding

        }
        question,answer,score = sess.run([cnn.question,cnn.answer,cnn.score12],feed_dict)
        print question.shape,answer.shape
        print score 

