#!/usr/bin/env python
#encoding=utf-8

import tensorflow as tf
import numpy as np
class QA_CNN(object):
    def __init__(self,max_input_left,max_input_right,batch_size,vocab_size,embedding_size,filter_sizes,
        num_filters,dropout_keep_prob,embeddings = None,l2_reg_lambda = 0.0,is_Embedding_Needed = False,trainable = True):

        self.question = tf.placeholder(tf.int32,[None,max_input_left],name = 'input_question')
        self.answer = tf.placeholder(tf.int32,[None,max_input_right],name = 'input_answer')
        self.answer_negative = tf.placeholder(tf.int32,[None,max_input_right],name = 'input_right')
        self.dropout_keep_prob = dropout_keep_prob
        self.num_filters = num_filters
        self.embeddings = embeddings
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.l2_reg_lambda = l2_reg_lambda
        self.l2_reg_lambda_keep_prob = tf.placeholder(tf.float32, name = "dropout_keep_prob")
        self.para = []
        with tf.name_scope('embedding'):
            if is_Embedding_Needed:
                print "load embedding"
                W = tf.Variable(np.array(self.embeddings),name="W" ,dtype="float32",trainable = trainable)
            else:
                W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name="W",trainable = trainable)
            self.embedding_W = W
            self.para.append(self.embedding_W)
            self.embedded_chars_q = tf.expand_dims(tf.nn.embedding_lookup(W,self.question),-1)
            self.embedded_chars_a = tf.expand_dims(tf.nn.embedding_lookup(W,self.answer),-1)
            self.embedded_chars_a_neg = tf.expand_dims(tf.nn.embedding_lookup(W,self.answer_negative),-1)
        print self.embedded_chars_q
        print self.embedded_chars_a
        pooled_outputs_1 = []
        pooled_outputs_2 = []
        pooled_outputs_3 = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev = 0.1), name="W")
                b = tf.Variable(tf.constant(0.0, shape=[num_filters]), name="b")

                self.para.append(W)
                self.para.append(b)
                conv = tf.nn.conv2d(
                    self.embedded_chars_q,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="conv-1"
                )
                print conv
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu-1")
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, max_input_left - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="poll-1"
                )

                print pooled
                pooled_outputs_1.append(pooled)

                conv = tf.nn.conv2d(
                    self.embedded_chars_a,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="conv-2"
                )
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu-2")
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, max_input_right - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="poll-2"
                )
                pooled_outputs_2.append(pooled)

                conv = tf.nn.conv2d(
                    self.embedded_chars_a_neg,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="conv-3"
                )
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu-3")
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, max_input_right - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="poll-3"
                )
                pooled_outputs_3.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)
        pooled_reshape_1 = tf.reshape(tf.concat(3, pooled_outputs_1), [-1, num_filters_total]) 
        pooled_reshape_2 = tf.reshape(tf.concat(3, pooled_outputs_2), [-1, num_filters_total]) 
        pooled_reshape_3 = tf.reshape(tf.concat(3, pooled_outputs_3), [-1, num_filters_total])
        #dropout
        pooled_flat_1 = tf.nn.dropout(pooled_reshape_1, self.dropout_keep_prob)
        pooled_flat_2 = tf.nn.dropout(pooled_reshape_2, self.dropout_keep_prob)
        pooled_flat_3 = tf.nn.dropout(pooled_reshape_3, self.dropout_keep_prob)

        pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.mul(pooled_flat_1, pooled_flat_1), 1)) #计算向量长度Batch模式
        pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.mul(pooled_flat_2, pooled_flat_2), 1))
        pooled_len_3 = tf.sqrt(tf.reduce_sum(tf.mul(pooled_flat_3, pooled_flat_3), 1))
        pooled_mul_12 = tf.reduce_sum(tf.mul(pooled_flat_1, pooled_flat_2), 1) #计算向量的点乘Batch模式
        pooled_mul_13 = tf.reduce_sum(tf.mul(pooled_flat_1, pooled_flat_3), 1)
        
        l2_loss = tf.constant(0.0)
        for p in self.para:
            l2_loss += tf.nn.l2_loss(p)
        with tf.name_scope("output"):
            self.score12 = tf.div(pooled_mul_12, tf.mul(pooled_len_1, pooled_len_2), name="scores") #计算向量夹角Batch模式
            self.score13 = tf.div(pooled_mul_13, tf.mul(pooled_len_1, pooled_len_3))

        zero = tf.constant(0, shape=[batch_size], dtype=tf.float32)
        margin = tf.constant(0.05, shape=[batch_size], dtype=tf.float32)
        with tf.name_scope("loss"):
            self.losses = tf.maximum(zero, tf.sub(margin, tf.sub(self.score12, self.score13)))
            self.loss = tf.reduce_sum(self.losses) + l2_reg_lambda * l2_loss
            print('loss ', self.loss)
        # Accuracy
        with tf.name_scope("accuracy"):
            self.correct = tf.equal(zero, self.losses)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct, "float"), name="accuracy")


if __name__ == '__main__':
    cnn = QA_CNN(max_input_left = 33,
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
        trainable = True)
    input_x_1 = np.reshape(np.arange(3 * 33),[3,33])
    input_x_2 = np.reshape(np.arange(3 * 40),[3,40])
    input_x_3 = np.reshape(np.arange(3 * 40),[3,40])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        question,answer = sess.run([cnn.embedded_chars_q,cnn.embedded_chars_a],feed_dict = {cnn.question:input_x_1,cnn.answer:input_x_2,
            cnn.answer_negative:input_x_3})
        print question.shape,answer.shape