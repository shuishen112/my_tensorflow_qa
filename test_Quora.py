import tensorflow as tf
import numpy as np
import os
import time
import datetime
from data_helpers import batch_gen_with_pair_overlap,batch_gen_with_pair_dns,dns_sample,load,prepare,batch_gen_with_pair,batch_gen_with_single,batch_gen_with_point_wise,getQAIndiceofTest,parseData,batch_gen_with_pair_whole
import operator
from QA_CNN_point_wise import QA
from QA_CNN_pair_wise import QA_CNN
from QA_CNN_extend import QA_CNN_extend
from attentive_pooling_network_test import QA_attentive
import random
import evaluation
import cPickle as pickle
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
tf.flags.DEFINE_string("loss","point_wise","loss function (default:point_wise)")
# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_boolean("trainable", True, "is embedding trainable? (default: False)")
tf.flags.DEFINE_integer("num_epochs", 500, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_boolean('overlap_needed',False,"is overlap used")
tf.flags.DEFINE_boolean('dns','False','whether use dns or not')
tf.flags.DEFINE_string('data','wiki','data set')
tf.flags.DEFINE_string('CNN_type','apn','data set')
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
print FLAGS.__flags
log_dir = 'log/'+ timeDay
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
data_file = log_dir + '/test_' + FLAGS.data + timeStamp
precision = data_file + 'precise'

@log_time_delta
def predict(sess,cnn,test,alphabet,batch_size,q_len,a_len):
    scores = []
    for data in batch_gen_with_single(test,alphabet,batch_size,q_len,a_len): 
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
def parser_data():
    
    df = pd.read_csv("data/quora/train.csv").fillna("")
    train = df[['question1','question2','is_duplicate']]
    train.columns = ['question','answer','flag']
    dftest = pd.read_csv("data/quora/test.csv").fillna("")
    test = dftest[['question1','question2',]]
    test.columns = ['question','answer']
    return train,test
def get_train_test(rate = 0.7):
    df,_ = parser_data()
    # shuffle the data
    size = len(df)
    flags = [True] * int(size * rate) +  [False] * (size - int(size * rate))
    random.seed(822)
    random.shuffle(flags)
    train = df[flags]
    test = df[map(lambda x:not x,flags)]
    return train,test
def test_quora(dns = False):
    train,test = parser_data()
    # train = train[:1000]
    # test = test[:1000]
    q_max_sent_length = max(map(lambda x:len(x),train['question'].str.split()))
    a_max_sent_length = max(map(lambda x:len(x),train['answer'].str.split()))
    print 'q_question_length:{} a_question_length:{}'.format(q_max_sent_length,a_max_sent_length)
    print 'train question unique:{}'.format(len(train['question'].unique()))
    print 'train length',len(train)
    print 'test length', len(test)
    alphabet,embeddings = prepare([train,test],is_embedding_needed = True,fresh = False)
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
                trainable = FLAGS.trainable,
                is_overlap = FLAGS.overlap_needed)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable = False)
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            saver = tf.train.Saver(tf.all_variables(), max_to_keep=20)
            model_file = 'runs/20170503/quora.model__0.115597'
            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            loadfile = model_file
            saver.restore(sess, loadfile)
            # seq_process(train, alphabet)
            # seq_process(test, alphabet)
            predicted = predict(sess,cnn,test,alphabet,FLAGS.batch_size,q_max_sent_length,a_max_sent_length)
            y_submission = predicted[:,-1]
            testid = np.arange(len(test))
            submission = pd.DataFrame({'a': testid, 'is_duplicate': y_submission})
            print submission.head()
            submission.to_csv("submission.csv", index=False)
            min_loss = 1
            for i in range(25):

                print "epoch :{} begins".format(i)
                if FLAGS.overlap_needed == False:

                    for x_left_batch, x_right_batch, y_batch in batch_gen_with_point_wise(train,alphabet,FLAGS.batch_size,overlap = FLAGS.overlap_needed,
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
                        if loss  < min_loss:
                            min_loss = loss
                            model = model_file +'__'+ str(loss)
                            save_path = saver.save(sess, model)
                            print "Model saved in file: ", save_path
                        print("{}:epoch {}: step {}, loss {:g}, acc {:g}  ".format(time_str, i,step, loss, accuracy))


                else:
                    for x_left_batch, x_right_batch, y_batch ,overlap in batch_gen_with_point_wise(train,alphabet,FLAGS.batch_size,overlap = FLAGS.overlap_needed,
                        q_len = q_max_sent_length,a_len = a_max_sent_length):
                        feed_dict = {
                            cnn.question: x_left_batch,
                            cnn.answer: x_right_batch,
                            cnn.overlap:overlap,
                            cnn.input_y: y_batch
                        }
                        _, step,loss, accuracy,pred ,scores = sess.run(
                        [train_op, global_step,cnn.loss, cnn.accuracy,cnn.predictions,cnn.scores],
                        feed_dict)
                        time_str = datetime.datetime.now().isoformat()
                       
                        print("{}: step {}, loss {:g}, acc {:g}  ".format(time_str, step, loss, accuracy))
            predicted = predict(sess,cnn,test,alphabet,FLAGS.batch_size,q_max_sent_length,a_max_sent_length)
            y_submission = predicted[:,-1]
            testid = np.arange(len(test))
            submission = pd.DataFrame({'a': testid, 'is_duplicate': y_submission})
            print submission.head()
            submission.to_csv("submission.csv", index=False)