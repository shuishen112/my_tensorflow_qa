#coding=utf-8
#! /usr/bin/env python3.4
import tensorflow as tf
import numpy as np
import os
import time
import datetime
from data_helpers import load,prepare,batch_gen_with_pair,batch_gen_with_single,batch_gen_with_point_wise,getQAIndiceofTest,parseData,batch_gen_with_pair_whole
import operator
from QA import QA
from qa_overlap import QA_overlap
from attention_pooling_qa import QA_CNN
import random
import evaluation
import matplotlib.pyplot as plt
import seaborn as sns
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

now = int(time.time()) 
    
timeArray = time.localtime(now)
timeStamp = time.strftime("%Y%m%d%H%M%S", timeArray)
print (timeStamp)
precision = 'log/test'+timeStamp

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 50, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 1, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.000001, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 0.000001, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_integer("max_len_left", 40, "max document length of left input")
tf.flags.DEFINE_integer("max_len_right", 40, "max document length of right input")
# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 500, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

def predict(sess,cnn,test,alphabet,batch_size):
    scores=[]
    for x_left_batch, x_right_batch in batch_gen_with_single(test,alphabet,batch_size):       
        feed_dict = {
                        cnn.question: x_left_batch,
                        cnn.answer: x_right_batch,
                        cnn.answer_negative: x_right_batch, 
                    }
        score = sess.run(cnn.score13, feed_dict)
        scores.extend(score)
    return scores[:len(test)]
def prediction(sess,cnn,test,alphabet,q_len,a_len):
    question,answer,overlap = parseData(test,alphabet,q_len,a_len)
    feed_dict = {
        cnn.question:question,
        cnn.answer:answer
    }
    score = sess.run(cnn.scores,feed_dict)
    return score
def test():
    train,test,dev = load("trec",filter = False)
    q_max_sent_length = max(map(lambda x:len(x),train['question'].str.split()))
    a_max_sent_length = max(map(lambda x:len(x),train['answer'].str.split()))
    print 'train length',len(train)
    print 'test length', len(test)
    print 'dev length', len(dev)
    alphabet = prepare([train,test,dev],is_embedding_needed = False)
    print 'alphabet:',len(alphabet)
    with tf.Graph().as_default():
        # with tf.device("/cpu:0"):
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default(),open("precision","w") as log:

            # train,test,dev = load("trec",filter=True)
            # alphabet,embeddings = prepare([train,test,dev],is_embedding_needed = True)
            cnn = QA(
                max_len_left = q_max_sent_length,
                max_len_right = a_max_sent_length,
                vocab_size = len(alphabet),
                embedding_size = FLAGS.embedding_dim,
                batch_size = FLAGS.batch_size,
                embeddings = None,
                dropout_keep_prob = FLAGS.dropout_keep_prob,
                filter_sizes = list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters = FLAGS.num_filters,
                l2_reg_lambda = FLAGS.l2_reg_lambda,
                is_Embedding_Needed = False)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            saver = tf.train.Saver(tf.all_variables(), max_to_keep=20)
            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # seq_process(train, alphabet)
            # seq_process(test, alphabet)
            for i in range(25):
                for x_left_batch, x_right_batch, y_batch in batch_gen_with_point_wise(train,alphabet,FLAGS.batch_size,overlap = False,
                    q_len = q_max_sent_length,a_len = a_max_sent_length):
                    feed_dict = {
                        cnn.question: x_left_batch,
                        cnn.answer: x_right_batch,
                        cnn.input_y: y_batch
                    }
                    _, step,loss, accuracy,pred ,scores = sess.run(
                    [train_op, global_step,cnn.loss, cnn.accuracy,cnn.predictions,cnn.scores],
                    feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                   
                    print("{}: step {}, loss {:g}, acc {:g}  ".format(time_str, step, loss, accuracy))
                    # print loss

                predicted = prediction(sess,cnn,test,alphabet,q_max_sent_length,a_max_sent_length)
                predicted_dev = prediction(sess,cnn,dev,alphabet,q_max_sent_length,a_max_sent_length)
                predicted_train = prediction(sess,cnn,train,alphabet,q_max_sent_length,a_max_sent_length)
                map_mrr_dev = evaluation.evaluationBypandas(dev,predicted_dev[:,-1])
                map_mrr_test = evaluation.evaluationBypandas(test,predicted[:,-1])
                print evaluation.evaluationBypandas(train,predicted_train[:,-1])
                print map_mrr_dev
                print map_mrr_test
                line = " {}: epoch: precision {}".format(i,map_mrr_test)
                log.write(line + '\n')
if __name__ == '__main__':
    test()