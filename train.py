#coding=utf-8
#! /usr/bin/env python3.4
import tensorflow as tf
import numpy as np
import os
import time
import datetime
from data_helpers import get_overlap_dict,replace_number,sample_data,batch_gen_with_pair_overlap,batch_gen_with_pair_dns,dns_sample,load,prepare,batch_gen_with_pair,batch_gen_with_single,batch_gen_with_point_wise,getQAIndiceofTest,batch_gen_with_pair_whole
import operator
from QA_CNN_point_wise import QA
from QA_CNN_pair_wise import QA_CNN
from QA_CNN_extend import QA_CNN_extend
from attentive_pooling_network_test import QA_attentive
import random
import evaluation
import cPickle as pickle
from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

now = int(time.time()) 
    
timeArray = time.localtime(now)
timeStamp = time.strftime("%Y%m%d%H%M%S", timeArray)
timeDay = time.strftime("%Y%m%d", timeArray)
print (timeStamp)

from functools import wraps
#print( tf.__version__)
def log_time_delta(func):
    @wraps(func)
    def _deco(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        delta = end - start
        print( "%s runed %.2f seconds"% (func.__name__,delta))
        return ret
    return _deco



# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim",300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 1, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.000001, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 1e-3, "learn rate( default: 0.0)")
tf.flags.DEFINE_integer("max_len_left", 40, "max document length of left input")
tf.flags.DEFINE_integer("max_len_right", 40, "max document length of right input")
tf.flags.DEFINE_string("loss","pair_wise","loss function (default:point_wise)")
tf.flags.DEFINE_integer('extend_feature_dim',10,'overlap_feature_dim')
# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_boolean("trainable", False, "is embedding trainable? (default: False)")
tf.flags.DEFINE_integer("num_epochs", 500, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_boolean('overlap_needed',False,"is overlap used")
tf.flags.DEFINE_boolean('dns','False','whether use dns or not')
tf.flags.DEFINE_string('data','nlpcc','data set')
tf.flags.DEFINE_string('CNN_type','qacnn','data set')
tf.flags.DEFINE_float('sample_train',1,'sampe my train data')
tf.flags.DEFINE_boolean('fresh',True,'wheather recalculate the embedding or overlap default is True')
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print(("{}={}".format(attr.upper(), value)))
log_dir = 'log/'+ timeDay
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
data_file = log_dir + '/test_' + FLAGS.data + timeStamp
precision = data_file + 'precise'

@log_time_delta
def predict(sess,cnn,test,alphabet,batch_size,q_len,a_len):
    scores = []
    d = get_overlap_dict(test,alphabet,q_len,a_len)
    for data in batch_gen_with_single(test,alphabet,batch_size,q_len,a_len,overlap_dict = d): 
        if FLAGS.loss ==  'point_wise':
            feed_dict = {
                cnn.question: data[0],
                cnn.answer: data[1]
            }
            score = sess.run(cnn.scores,feed_dict)
        else:
            feed_dict = {
                        cnn.question: data[0],
                        cnn.answer: data[1],
                        cnn.q_pos_overlap: data[2],
                        cnn.a_pos_overlap: data[3]
                    }
            score = sess.run(cnn.score12, feed_dict)
        scores.extend(score)
    return np.array(scores[:len(test)])
@log_time_delta
def test_pair_wise(dns = FLAGS.dns):
    train,test,dev = load(FLAGS.data,filter = False)
    train = train.fillna('')
    test = test.fillna('')
    dev = dev.fillna('')
    # train = train[:1000]
    # test = test[:1000]
    # dev = dev[:1000]
    # submit = submit[:1000]
    q_max_sent_length = max(map(lambda x:len(x),train['question'].str.split()))
    a_max_sent_length = max(map(lambda x:len(x),train['answer'].str.split()))
    print 'q_question_length:{} a_question_length:{}'.format(q_max_sent_length,a_max_sent_length)
    print 'train question unique:{}'.format(len(train['question'].unique()))
    print 'train length',len(train)
    print 'test length', len(test)
    print 'dev length', len(dev)
    alphabet,embeddings = prepare([train,test,dev],dim = FLAGS.embedding_dim,is_embedding_needed = True,fresh = FLAGS.fresh)
    # alphabet,embeddings = prepare_300([train,test,dev])
    print 'alphabet:',len(alphabet)
    with tf.Graph().as_default(), tf.device("/gpu:0"):
        # with tf.device("/cpu:0"):
        session_conf = tf.ConfigProto()
        session_conf.allow_soft_placement = FLAGS.allow_soft_placement
        session_conf.log_device_placement = FLAGS.log_device_placement
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default(),open(precision,"w") as log:
            log.write(str(FLAGS.__flags) + '\n')
            # train,test,dev = load("trec",filter=True)
            # alphabet,embeddings = prepare([train,test,dev],is_embedding_needed = True)
            print "start build model"
            cnn = QA_CNN_extend(
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
                overlap_needed = FLAGS.overlap_needed,
                learning_rate=FLAGS.learning_rate,
                trainable = FLAGS.trainable,
                extend_feature_dim = FLAGS.extend_feature_dim,
                model_type=FLAGS.CNN_type)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable = False)
            optimizer = tf.train.AdamOptimizer(cnn.learning_rate)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step = global_step)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
            # Initialize all variables
            print "build over"
            sess.run(tf.global_variables_initializer())
            print "variables_initializer"
            if dns == True:
                loadfile="runs/20170604/20170604183633__nlpcc0.833940715393"
                saver.restore(sess, loadfile)
                predicted = predict(sess,cnn,train,alphabet,FLAGS.batch_size,q_max_sent_length,a_max_sent_length)
                map_mrr_train = evaluation.evaluationBypandas(train,predicted)
                predicted = predict(sess,cnn,test,alphabet,FLAGS.batch_size,q_max_sent_length,a_max_sent_length)
                map_mrr_test = evaluation.evaluationBypandas(test,predicted)
                print map_mrr_train
                print map_mrr_test
            # seq_process(train, alphabet)
            # seq_process(test, alphabet)
            '''
            print 'get my submit result'
            loadfile="runs/20170604/20170604183633__nlpcc0.833940715393"
            saver.restore(sess, loadfile)
            predicted = predict(sess,cnn,train,alphabet,FLAGS.batch_size,q_max_sent_length,a_max_sent_length)
            train['predicted'] = predicted
            train['predicted'].to_csv('train.QApair.TJU_IR_QA2017_train.score',index = False,sep = '\t')
            map_mrr_train = evaluation.evaluationBypandas(train,predicted)
            predicted_test = predict(sess,cnn,test,alphabet,FLAGS.batch_size,q_max_sent_length,a_max_sent_length)
            test['predicted'] = predicted_test
            test['predicted'].to_csv('train.QApair.TJU_IR_QA2017.score',index = False,sep = '\t')
            map_mrr_test = evaluation.evaluationBypandas(test,predicted_test)
            print 'map_mrr train',map_mrr_train
            print 'map_prr dev',map_mrr_test

            predict_submit = predict(sess,cnn,submit,alphabet,FLAGS.batch_size,q_max_sent_length,a_max_sent_length)
            submit['predicted'] = predict_submit
            submit['predicted'].to_csv('train.QApair.TJU_IR_QA2017_submit.score',index = False,sep = '\t')
            print 'predict over'

            '''
            map_max = 0.65
            for i in range(1000):
                if dns ==True:
                    samples = dns_sample(train,alphabet,q_max_sent_length,
                        a_max_sent_length,sess,cnn,FLAGS.batch_size,neg_sample_num = 10)
                    datas = batch_gen_with_pair_dns(samples,FLAGS.batch_size)
                else:
                    d = get_overlap_dict(train,alphabet,q_len = q_max_sent_length,a_len = a_max_sent_length)
                    datas = batch_gen_with_pair_overlap(train,alphabet,FLAGS.batch_size,
                        q_len = q_max_sent_length,a_len = a_max_sent_length,fresh = FLAGS.fresh,overlap_dict = d)        
                print "load data"
                for data in datas:
                    feed_dict = {
                        cnn.question: data[0],
                        cnn.answer: data[1],
                        cnn.answer_negative:data[2],
                        cnn.q_pos_overlap:data[3],
                        cnn.q_neg_overlap:data[4],
                        cnn.a_pos_overlap:data[5],
                        cnn.a_neg_overlap:data[6]
                    }
                    _, step,loss, accuracy,score12,score13 = sess.run(
                    [train_op, global_step,cnn.loss, cnn.accuracy,cnn.score12,cnn.score13],
                    feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                   
                    print("{}: step {}, loss {:g}, acc {:g} ,positive {:g},negative {:g}".format(time_str, step, loss, accuracy,np.mean(score12),np.mean(score13)))
                    line = "{}: step {}, loss {:g}, acc {:g} ,positive {:g},negative {:g}".format(time_str, step, loss, accuracy,np.mean(score12),np.mean(score13))
                    # print loss
                if i % 1 == 0:
                    predicted = predict(sess,cnn,test,alphabet,FLAGS.batch_size,q_max_sent_length,a_max_sent_length)
                    map_mrr_test = evaluation.evaluationBypandas(test,predicted)
                    print "{}:epoch:test map mrr {}".format(i,map_mrr_test)
                    line = " {}:epoch: map_test{}".format(i,map_mrr_test[0])
                    if map_mrr_test[0] > map_max:
                        map_max = map_mrr_test[0]
                        timeStamp = time.strftime("%Y%m%d%H%M%S", time.localtime(int(time.time())))
                        folder = 'runs/' + timeDay
                        out_dir = folder +'/'+timeStamp+'__'+FLAGS.data+str(map_mrr_test[0])
                        if not os.path.exists(folder):
                            os.makedirs(folder)
                        save_path = saver.save(sess, out_dir)
                        print "Model saved in file: ", save_path

                '''
                predicted = predict(sess,cnn,train,alphabet,FLAGS.batch_size,q_max_sent_length,a_max_sent_length)
                map_mrr_train = evaluation.evaluationBypandas(train,predicted)
                predicted = predict(sess,cnn,dev,alphabet,FLAGS.batch_size,q_max_sent_length,a_max_sent_length)
                map_mrr_dev = evaluation.evaluationBypandas(dev,predicted)
                print "{}:epoch:train map mrr {}".format(i,map_mrr_train)
                # print "{}:epoch:test map mrr {}".format(i,map_mrr_test)
                print "{}:epoch:dev map mrr {}".format(i,map_mrr_dev)
                if map_mrr_dev[0] > map_max:
                    map_max = map_mrr_dev[0]
                    predicted = predict(sess,cnn,test,alphabet,FLAGS.batch_size,q_max_sent_length,a_max_sent_length)
                    map_mrr_test = evaluation.evaluationBypandas(test,predicted)
                    print "{}:epoch:test map mrr {}".format(i,map_mrr_test)
                    line = " {}:epoch: map_test{}".format(i,map_mrr_test[0])
                    if map_mrr_test[0] > map_max:
                        timeStamp = time.strftime("%Y%m%d%H%M%S", time.localtime(int(time.time())))
                        folder = 'runs/' + timeDay
                        out_dir = folder +'/'+timeStamp+'__'+FLAGS.data+str(map_mrr_test[0])
                        if not os.path.exists(folder):
                            os.makedirs(folder)
                        save_path = saver.save(sess, out_dir)
                        print "Model saved in file: ", save_path
                        
                '''
                # # predicted_train = prediction(sess,cnn,train,alphabet,q_max_sent_length,a_max_sent_length)
                # map_mrr_dev = evaluation.evaluationBypandas(dev,predicted_dev[:,-1])
                # map_mrr_test = evaluation.evaluationBypandas(test,predicted[:,-1])
                # map_mrr_train = evaluation.evaluationBypandas(train,predicted_train[:,-1])
                # # print evaluation.evaluationBypandas(train,predicted_train[:,-1])
                
                # line = " {}:epoch: map_train{}----map_test{}----map_dev{}".format(i,map_mrr_train[0],map_mrr_test[0],map_mrr_dev[0])
                # line = " {}:epoch: map_train{}----map_dev{}".format(i,map_mrr_train[0],map_mrr_dev[0])
                log.write(line + '\n')
                log.flush()
                

if __name__ == '__main__':
    # test_quora()
    if FLAGS.loss == 'point_wise':
        test_point_wise()
    else:
        test_pair_wise()
    # test_pair_wise()
    # test_point_wise()
