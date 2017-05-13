import pandas as pd
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