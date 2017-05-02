#!/usr/bin/env python
#encoding=utf-8

import tensorflow as tf
import numpy as np

class QA(object):
    def __init__(
      self, max_len_left, max_len_right, vocab_size,embedding_size,batch_size,
      embeddings,dropout_keep_prob,filter_sizes, num_filters,l2_reg_lambda = 0.0, is_Embedding_Needed = False,trainable=True):

        self.question = tf.placeholder(tf.int32,[None,max_len_left],name = 'input_question')
        self.answer = tf.placeholder(tf.int32,[None,max_len_right],name = 'input_answer')
        self.input_y = tf.placeholder(tf.float32, [None,2], name = "input_y")
        self.dropout_keep_prob = dropout_keep_prob
        self.num_filters = num_filters
        self.embeddings = embeddings
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.l2_reg_lambda = l2_reg_lambda

        self.para = []
        # Embedding layer for both CNN
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            if is_Embedding_Needed:
                print "load embedding"
                W = tf.Variable(np.array(embeddings),name="W" ,dtype="float32",trainable = trainable )
            else:
                W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name="W",trainable = trainable)
            self.embedding_W = W
            self.embedded_chars_q = tf.expand_dims(tf.nn.embedding_lookup(W,self.question),-1)
            self.embedded_chars_a = tf.expand_dims(tf.nn.embedding_lookup(W,self.answer),-1)
            self.para.append(self.embedding_W)
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs_left = []
        pooled_outputs_right = []
        for i, filter_size in enumerate(filter_sizes):
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            # Convolution Layer
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape = [num_filters]), name="b")
            self.para.append(W)
            self.para.append(b)
            
            with tf.name_scope("conv-maxpool-left-%s" % filter_size):               
                conv = tf.nn.conv2d(
                    self.embedded_chars_q,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, max_len_left - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs_left.append(pooled)
            with tf.name_scope("conv-maxpool-right-%s" % filter_size):
                conv = tf.nn.conv2d(
                    self.embedded_chars_a,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, max_len_right - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs_right.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool_left = tf.reshape(tf.concat(3, pooled_outputs_left), [-1, num_filters_total], name='h_pool_left')
        self.h_pool_right = tf.reshape(tf.concat(3, pooled_outputs_right), [-1, num_filters_total], name='h_pool_right')
        print self.h_pool_left
        print self.h_pool_right
        l2_loss = tf.constant(0.0)
        # Make input for classification
        self.new_input = tf.concat(1, [self.h_pool_left,self.h_pool_right], name='new_input')

        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.new_input, self.dropout_keep_prob,name = 'drop_out')

        with tf.name_scope("output"):
            W = tf.get_variable(
                "W_output",
                shape=[2 * num_filters_total, 2],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")
            self.para.append(W)
            self.para.append(b)

            for p in self.para:
                l2_loss += tf.nn.l2_loss(p)            
            self.scores = tf.nn.softmax(tf.nn.xw_plus_b(self.h_drop, W, b, name="scores"))
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            # losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            losses = -tf.reduce_sum(self.input_y * tf.log(self.scores),reduction_indices = 1)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
       
