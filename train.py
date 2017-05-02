#coding=utf-8
#! /usr/bin/env python3.4
import tensorflow as tf
import numpy as np
import os
import time
import datetime
from data_helpers import batch_gen_with_pair_dns,dns_sample,prepare_300,load,prepare,batch_gen_with_pair,batch_gen_with_single,batch_gen_with_point_wise,getQAIndiceofTest,parseData,batch_gen_with_pair_whole
import operator
from QA import QA
from QA_CNN import QA_CNN
import random
import evaluation
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

now = int(time.time()) 
    
timeArray = time.localtime(now)
timeStamp = time.strftime("%Y%m%d%H%M%S", timeArray)
print (timeStamp)


# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim",50, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "2,3,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 1, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.000001, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 1e-3, "learn rate( default: 0.0)")
tf.flags.DEFINE_integer("max_len_left", 40, "max document length of left input")
tf.flags.DEFINE_integer("max_len_right", 40, "max document length of right input")
tf.flags.DEFINE_string("loss","pair_wise","loss function (default:point_wise)")
# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_boolean("trainable", True, "is embedding trainable? (default: False)")
tf.flags.DEFINE_integer("num_epochs", 500, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
precision = 'log/test_wiki'+timeStamp+'embedding__trainable'+str(FLAGS.trainable)+FLAGS.loss
def predict(sess,cnn,test,alphabet,batch_size,q_len,a_len):
    scores=[]
    for x_left_batch, x_right_batch in batch_gen_with_single(test,alphabet,batch_size,q_len,a_len):       
        
        if FLAGS.loss == 'point_wise':
            feed_dict = {
                        cnn.question: x_left_batch,
                        cnn.answer: x_right_batch,
                    }
            score = sess.run(cnn.scores, feed_dict)
        else:
            feed_dict = {
                        cnn.question: x_left_batch,
                        cnn.answer: x_right_batch,
                        cnn.answer_negative: x_right_batch
                    }
            score = sess.run(cnn.score13, feed_dict)
        scores.extend(score)
    return np.array(scores[:len(test)])
def prediction(sess,cnn,test,alphabet,q_len,a_len):
    question,answer,overlap = parseData(test,alphabet,q_len,a_len)
    feed_dict = {
        cnn.question:question,
        cnn.answer:answer
    }
    score = sess.run(cnn.scores,feed_dict)
    return score
def test_point_wise():
    train,test,dev = load("wiki",filter = True)
    q_max_sent_length = max(map(lambda x:len(x),train['question'].str.split()))
    a_max_sent_length = max(map(lambda x:len(x),train['answer'].str.split()))
    print 'train question unique:{}'.format(len(train['question'].unique()))
    print 'train length',len(train)
    print 'test length', len(test)
    print 'dev length', len(dev)
    alphabet,embeddings = prepare([train,test,dev],is_embedding_needed = True)
    print 'alphabet:',len(alphabet)
    with tf.Graph().as_default():
        # with tf.device("/cpu:0"):
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default(),open(precision,"w") as log:

            # train,test,dev = load("trec",filter=True)
            # alphabet,embeddings = prepare([train,test,dev],is_embedding_needed = True)
            cnn = QA(
                max_len_left = q_max_sent_length,
                max_len_right = a_max_sent_length,
                vocab_size = len(alphabet),
                embedding_size = FLAGS.embedding_dim,
                batch_size = FLAGS.batch_size,
                embeddings = embeddings,
                dropout_keep_prob = FLAGS.dropout_keep_prob,
                filter_sizes = list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters = FLAGS.num_filters,
                l2_reg_lambda = FLAGS.l2_reg_lambda,
                is_Embedding_Needed = True,
                trainable = FLAGS.trainable)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable = False)
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            saver = tf.train.Saver(tf.all_variables(), max_to_keep=20)
            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # seq_process(train, alphabet)
            # seq_process(test, alphabet)
            for i in range(100):
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
                predicted_train = predict(sess,cnn,train,alphabet,FLAGS.batch_size,q_max_sent_length,a_max_sent_length)
                predicted = prediction(sess,cnn,test,alphabet,q_max_sent_length,a_max_sent_length)
                predicted_dev = prediction(sess,cnn,dev,alphabet,q_max_sent_length,a_max_sent_length)
                # predicted_train = prediction(sess,cnn,train,alphabet,q_max_sent_length,a_max_sent_length)
                map_mrr_dev = evaluation.evaluationBypandas(dev,predicted_dev[:,-1])
                map_mrr_test = evaluation.evaluationBypandas(test,predicted[:,-1])
                map_mrr_train = evaluation.evaluationBypandas(train,predicted_train[:,-1])
                # print evaluation.evaluationBypandas(train,predicted_train[:,-1])
                print "{}:epoch:map mrr {}".format(i,map_mrr_train)
                print "{}:epoch:map mrr {}".format(i,map_mrr_test)
                print "{}:epoch:map mrr {}".format(i,map_mrr_dev)
                line = " {}: epoch: precision {}".format(i,map_mrr_test)
                log.write(line + '\n')
                log.flush()
            log.close()
def test_pair_wise():
    train,test,dev = load("wiki",filter = True)
    q_max_sent_length = max(map(lambda x:len(x),train['question'].str.split()))
    a_max_sent_length = max(map(lambda x:len(x),train['answer'].str.split()))
    print 'q_question_length:{} a_question_length:{}'.format(q_max_sent_length,a_max_sent_length)
    print 'train question unique:{}'.format(len(train['question'].unique()))
    print 'train length',len(train)
    print 'test length', len(test)
    print 'dev length', len(dev)
    alphabet,embeddings = prepare([train,test,dev],is_embedding_needed = True,fresh = True)
    # alphabet,embeddings = prepare_300([train,test,dev])
    print 'alphabet:',len(alphabet)

    with tf.Graph().as_default():
        # with tf.device("/cpu:0"):
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default(),open(precision,"w") as log:

            # train,test,dev = load("trec",filter=True)
            # alphabet,embeddings = prepare([train,test,dev],is_embedding_needed = True)
            cnn = QA_CNN(
                max_input_left = q_max_sent_length,
                max_input_right = a_max_sent_length,
                batch_size = FLAGS.batch_size,
                vocab_size = len(alphabet),
                embedding_size = FLAGS.embedding_dim,
                filter_sizes = list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters = FLAGS.num_filters,                
                dropout_keep_prob = FLAGS.dropout_keep_prob,
                embeddings = embeddings,                
                l2_reg_lambda = FLAGS.l2_reg_lambda,
                is_Embedding_Needed = True,
                trainable = FLAGS.trainable)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable = False)
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # seq_process(train, alphabet)
            # seq_process(test, alphabet)
            for i in range(100):

                samples = dns_sample(train,alphabet,q_max_sent_length,
                    a_max_sent_length,sess,cnn,FLAGS.batch_size,neg_sample_num = 30)

                # for x_batch_1, x_batch_2, x_batch_3 in batch_gen_with_pair(train,alphabet,FLAGS.batch_size,
                #     q_len = q_max_sent_length,a_len = a_max_sent_length):
                for x_batch_1, x_batch_2, x_batch_3 in batch_gen_with_pair_dns(samples,FLAGS.batch_size):
                    feed_dict = {
                        cnn.question: x_batch_1,
                        cnn.answer: x_batch_2,
                        cnn.answer_negative:x_batch_3
                    }
                    _, step,loss, accuracy,score12,score13 = sess.run(
                    [train_op, global_step,cnn.loss, cnn.accuracy,cnn.score12,cnn.score13],
                    feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                   
                    print("{}: step {}, loss {:g}, acc {:g} ,positive {:g},negative {:g}".format(time_str, step, loss, accuracy,np.mean(score12),np.mean(score13)))
                    # print loss
                # predicted = predict(sess,cnn,train,alphabet,FLAGS.batch_size,q_max_sent_length,a_max_sent_length)
                # map_mrr_train = evaluation.evaluationBypandas(train,predicted)
                predicted = predict(sess,cnn,test,alphabet,FLAGS.batch_size,q_max_sent_length,a_max_sent_length)
                map_mrr_test = evaluation.evaluationBypandas(test,predicted)
                # # predicted_train = prediction(sess,cnn,train,alphabet,q_max_sent_length,a_max_sent_length)
                # map_mrr_dev = evaluation.evaluationBypandas(dev,predicted_dev[:,-1])
                # map_mrr_test = evaluation.evaluationBypandas(test,predicted[:,-1])
                # map_mrr_train = evaluation.evaluationBypandas(train,predicted_train[:,-1])
                # # print evaluation.evaluationBypandas(train,predicted_train[:,-1])
                # print "{}:epoch:train map mrr {}".format(i,map_mrr_train)
                print "{}:epoch:test map mrr {}".format(i,map_mrr_test)
                # print "{}:epoch:map mrr {}".format(i,map_mrr_dev)
                line = " {}:epoch: ----map_mrr_test{}".format(i,map_mrr_test)
                log.write(line + '\n')
                log.flush()
            log.close()
def test_dns():
    train,test,dev = load("wiki",filter = True)
    q_max_sent_length = max(map(lambda x:len(x),train['question'].str.split()))
    a_max_sent_length = max(map(lambda x:len(x),train['answer'].str.split()))
    print 'q_question_length:{} a_question_length:{}'.format(q_max_sent_length,a_max_sent_length)
    print 'train question unique:{}'.format(len(train['question'].unique()))
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
        with sess.as_default(),open(precision,"w") as log:

            # train,test,dev = load("trec",filter=True)
            # alphabet,embeddings = prepare([train,test,dev],is_embedding_needed = True)
            cnn = QA_CNN(
                max_input_left = q_max_sent_length,
                max_input_right = a_max_sent_length,
                batch_size = FLAGS.batch_size,
                vocab_size = len(alphabet),
                embedding_size = FLAGS.embedding_dim,
                filter_sizes = list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters = FLAGS.num_filters,                
                dropout_keep_prob = FLAGS.dropout_keep_prob,
                embeddings = None,                
                l2_reg_lambda = FLAGS.l2_reg_lambda,
                is_Embedding_Needed = False,
                trainable = FLAGS.trainable)
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable = False)
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step = global_step)
            saver = tf.train.Saver(tf.all_variables(), max_to_keep=20)
            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            for i in range(100):
                samples = dns_sample(train,alphabet,q_max_sent_length,
                    a_max_sent_length,sess,cnn,FLAGS.batch_size,neg_sample_num = 30)
                for x_batch_1, x_batch_2, x_batch_3 in batch_gen_with_pair_dns(samples,FLAGS.batch_size):
                    print i
if __name__ == '__main__':
    # test()
    test_pair_wise()
    # test_dns()
    # test_point_wise()