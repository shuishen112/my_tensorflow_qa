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
# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 50, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 200, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 1, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.000001, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 0.0001, "L2 regularizaion lambda (default: 0.0)")
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
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")





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
        cnn.input_left:question,
        cnn.input_right:answer,
        cnn.overlap:overlap,
        cnn.dropout_keep_prob:FLAGS.dropout_keep_prob
    }

    score = sess.run(cnn.scores,feed_dict)
    return score

def prediction_2(sess,cnn,test,alphabet,q_len=40,a_len=40):
    question,answer,overlap = parseData(test,alphabet,q_len,a_len)
    feed_dict = {
        cnn.question: question,
        cnn.answer: answer,
        cnn.answer_negative: answer     
    }

    score = sess.run(cnn.score13,feed_dict)
    return score

def test():
    train,test,dev = load("trec",filter = False)
    q_max_sent_length = max(map(lambda x:len(x),train['question'].str.split()))
    a_max_sent_length = max(map(lambda x:len(x),train['answer'].str.split()))

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
        with sess.as_default(),open("precision.txt","w") as log:

            # train,test,dev = load("trec",filter=True)
            # alphabet,embeddings = prepare([train,test,dev],is_embedding_needed = True)
            cnn = QA_overlap(
                max_len_left = q_max_sent_length,
                max_len_right = a_max_sent_length,
                vocab_size = len(alphabet),
                embedding_size=FLAGS.embedding_dim,
                embeddings = embeddings,
                # filter_sizes = list(map(int, FLAGS.filter_sizes.split(","))),
                filter_sizes = [5],
                num_filters = FLAGS.num_filters,
                num_hidden = 10,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                is_Embedding_Needed = True)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            saver = tf.train.Saver(tf.all_variables(), max_to_keep=20)
            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            #summary
            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.merge_summary(grad_summaries)
            # Output directory for models and summaries

            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.scalar_summary("loss", cnn.loss)
            acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)
            
            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            # saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # seq_process(train, alphabet)
            # seq_process(test, alphabet)
            for i in range(25):
                for x_left_batch, x_right_batch, y_batch,overlap in batch_gen_with_point_wise(train,alphabet,FLAGS.batch_size,overlap = True,
                    q_len = q_max_sent_length,a_len = a_max_sent_length):
                    feed_dict = {
                        cnn.input_left: x_left_batch,
                        cnn.input_right: x_right_batch,
                        cnn.input_y: y_batch,
                        cnn.overlap:overlap,
                        cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                    }

                    _, step, summaries,loss, accuracy,pred ,scores = sess.run(
                    [train_op, global_step,train_summary_op,cnn.loss, cnn.accuracy,cnn.predictions,cnn.scores],
                    feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                   
                    print("{}: step {}, loss {:g}, acc {:g}  ".format(time_str, step, loss, accuracy))
                    train_summary_writer.add_summary(summaries, step)
                    # print loss

                predicted = prediction(sess,cnn,test,alphabet,q_max_sent_length,a_max_sent_length)
                predicted_dev = prediction(sess,cnn,dev,alphabet,q_max_sent_length,a_max_sent_length)
                # predicted_train = prediction(sess,cnn,train,alphabet)
                print np.array(predicted).shape
                print len(predicted)
                print len(test)   
                map_mrr_dev = evaluation.evaluationBypandas(dev,predicted_dev[:,-1])
                map_mrr_test = evaluation.evaluationBypandas(test,predicted[:,-1])
                # print evaluation.evaluationBypandas(train,predicted_train[:,-1])
                print map_mrr_dev
                print map_mrr_test
                line = " {}: epoch: precision {}".format(i,map_mrr_test)
                log.write(line + '\n')

def main():
    train,test,dev = load("wiki",filter = True)
    q_max_sent_length = max(map(lambda x:len(x),train['question'].str.split()))
    a_max_sent_length = max(map(lambda x:len(x),train['answer'].str.split()))
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
        with sess.as_default(),open("precision.txt","w") as log:
            cnn = QA_CNN(
                max_input_left = 40,
                max_input_right = 40,
                batch_size = FLAGS.batch_size,
                vocab_size=len(alphabet),
                embeddings = embeddings,
                embedding_size=FLAGS.embedding_dim,
                num_filter=FLAGS.num_filters,
                dropout_keep_prob = 1.0,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                is_embedding_needded = True)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            saver = tf.train.Saver(tf.all_variables(), max_to_keep=20)
            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            for i in range(1000):
                for x_batch_1, x_batch_2, x_batch_3 in batch_gen_with_pair_whole(train,alphabet,FLAGS.batch_size):

                    feed_dict = {
                        cnn.question: x_batch_1,
                        cnn.answer: x_batch_2,
                        cnn.answer_negative: x_batch_3, 
                    }

                    _, step,  loss, accuracy,scores1,scores2 ,a1,a2 ,U= sess.run( [train_op,global_step,cnn.loss,cnn.accuracy,cnn.score12,cnn.score13,cnn.attention_q,cnn.attention_a,cnn.U ], feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g} positive {:g} negative {:g} mean_pooling {:g}".format(time_str, step, loss, accuracy,np.mean(scores1),np.mean(scores2),np.mean(a1)))
                    # print a1
                predicted = predict(sess,cnn,train,alphabet,FLAGS.batch_size)
                print (evaluation.evaluationBypandas(train,predicted))
                predicted = predict(sess,cnn,test,alphabet,FLAGS.batch_size)
                print (evaluation.evaluationBypandas(test,predicted))
                
                   
                 
         

                # 
                

def theano_verion():

    train,test=load("wiki",filter=True)
    alphabet,embeddings=prepare([train,test])
    test_input=getQAIndiceofTest(test,alphabet)
    from model import Model1
    model = Model1(50, 50, 50, len(alphabet.keys()), embeddings)
    #pdb.set_trace()
    #print((model.predict([q_train, a_train])))
    # start training
    for epoch in range(20):
        
        for x_trainq, x_traina, y_train1 in  batch_gen_with_point_wise(train,alphabet,FLAGS.batch_size):
            loss, acc = model.train_on_batch([x_trainq, x_traina], y_train1)
            perf = str(loss) + " " + str(acc)
            print("loss is %f with acc %f"%(loss, acc))

        #y_train1 = y_train1.reshape(y_train1.shape[0],1)
        #x = model.predict([x_trainq, x_traina])
        predicted = model.predict_on_batch(test_input)
        print (evaluation.evaluationBypandas(test,predicted))


        evaluation.briany_test_file(test,predicted)
        print ("\n\n\n\n\n")

if __name__ == '__main__':
    # overlap_visualize()
    main()
    # test()