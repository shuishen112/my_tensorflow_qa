import numpy as np
import random,os,math
import pandas as pd
from gensim.models.keyedvectors import KeyedVectors
import sklearn
import multiprocessing
import time
import cPickle
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import evaluation
from features import overlap_jiabing
cores = multiprocessing.cpu_count()
dataset= "wiki"
UNKNOWN_WORD_IDX = 0

class Alphabet(dict):
    def __init__(self, start_feature_id=1):
        self.fid = start_feature_id

    def add(self, item):
        idx = self.get(item, None)
        if idx is None:
            idx = self.fid
            self[item] = idx
      # self[idx] = item
            self.fid += 1
        return idx

    def dump(self, fname):
        with open(fname, "w") as out:
            for k in sorted(self.keys()):
                out.write("{}\t{}\n".format(k, self[k]))

def load_bin_vec(fname="embedding/GoogleNews-vectors-negative300.bin"):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    # fname="embedding/aquaint+wiki.txt.gz.ndim=50.bin"
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        print ('vocab_size, layer1_size', vocab_size, layer1_size)
        for i, line in enumerate(range(vocab_size)):
            print (i)
            if i % 100000 == 0:
                print ('.',)
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')

        return word_vecs,layer1_size
# get data for lexdecomp
def load_bin_vec(fname,words):
    print fname
    vocab = set(words)
    word_vecs = {}
    voc = open('GoogleNews-vectors-300d.voc','w')
    embedding = []
    with open(fname,'rb') as f:
        header = f.readline()
        # convert the str into int:3000000 300
        vocab_size,layer1_size = map(int,header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size# float32.itemsize:4
        print 'vocab_size,layer1_size',vocab_size,layer1_size
        for i,line in enumerate(xrange(vocab_size)):
            if i % 100000 == 0:
                print 'epch %d' % i
            word = []
            while True:
                # the word is split by ' '
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            temp_word = word.decode('utf-8').lower()
            if temp_word in vocab:
                # the vector fo word is just like this:'\x01\x02'
                word_vecs[temp_word] = np.fromstring(f.read(binary_len),dtype = 'float32')
                voc.write(temp_word + '\n')
                embedding.append(word_vecs[temp_word])
            else:
                # print word
                f.read(binary_len)
        print 'done'
        print 'words found in wor2vec embedding ',len(word_vecs.keys())
        np.save('GoogleNews-vectors-300d',embedding)
        return word_vecs
def load_text_vec(filename="",embedding_size=100):
    vectors = {}
    for line in open(filename):
        items = line.strip().split(' ')
        if len(items) ==2:
            vocab_size, embedding_size= items[0],items[1]
            print ( vocab_size, embedding_size)
        else:
            vectors[items[0]] = items[1:]
    return vectors

def load_vectors( vectors,vocab,dim_size):
    if vocab==None:
        return
    embeddings=[]
    for word in vocab:
        embeddings.append(vectors.get(word,np.random.uniform(-1,1,dim_size).tolist() ))
    return embeddings
def encode_to_split(sentence,alphabet,max_sentence = 40):
    indices=[]
    tokens=sentence.lower().split()
    for word in tokens:
        indices.append(alphabet[str(word)])
    results=indices+[alphabet["END"]]*(max_sentence-len(indices))
    return results[:max_sentence]
def transform(flag):
    if flag == 1:
        return [0,1]
    else:
        return [1,0]

def batch_gen_with_single(df,alphabet, batch_size=10):
    pairs=[]
    for index,row in df.iterrows():
        quetion = encode_to_split(row["question"],alphabet)
        answer = encode_to_split(row["answer"],alphabet)
        pairs.append((quetion,answer))
    # n_batches= int(math.ceil(df["flag"].sum()*1.0/batch_size))
    n_batches= int(len(pairs)*1.0/batch_size)
    # pairs = sklearn.utils.shuffle(pairs,random_state =132)

    for i in range(0,n_batches):
        batch = pairs[i*batch_size:(i+1) * batch_size]

        yield ([pair[j] for pair in batch]  for j in range(2))
    batch= pairs[n_batches*batch_size:] + [pairs[n_batches*batch_size]] * (batch_size- len(pairs)+n_batches*batch_size  )
    yield ([pair[i] for pair in batch]  for i in range(2))
# this is for trec,wiki data
def parseData(df,alphabet,q_len = 33,a_len = 40):
    q = []
    a = []
    overlap = []
    for index,row in df.iterrows():
        question = encode_to_split(row["question"],alphabet,max_sentence = q_len)
        answer = encode_to_split(row["answer"],alphabet,max_sentence = a_len)
        lap = overlap_jiabing(row)
        q.append(question)
        a.append(answer)
        overlap.append(lap)
    return q,a,overlap
def batch_gen_with_point_wise(df,alphabet, batch_size=10,overlap = False,q_len = 33,a_len = 40):
    #inputq inputa intput_y overlap
    if overlap:
        input_num = 4
        print 'overlap is needed'
    else:
        input_num = 3
    pairs=[]
    for index,row in df.iterrows():
        question = encode_to_split(row["question"],alphabet,max_sentence = q_len)
        answer = encode_to_split(row["answer"],alphabet,max_sentence = a_len)
        label = transform(row["flag"])
        if overlap:
            lap = overlap_jiabing(row)
            pairs.append((question,answer,label,lap))
        else:
            pairs.append((question,answer,label))
    # n_batches= int(math.ceil(df["flag"].sum()*1.0/batch_size))
    n_batches= int(len(pairs)*1.0/batch_size)
    pairs = sklearn.utils.shuffle(pairs,random_state = 132)

    for i in range(0,n_batches):
        batch = pairs[i*batch_size:(i+1) * batch_size]
        yield (np.array([pair[i] for pair in batch])  for i in range(input_num))
    batch= pairs[n_batches*batch_size:] + [pairs[n_batches*batch_size]] * (batch_size- len(pairs)+n_batches*batch_size  )
    yield (np.array([pair[i] for pair in batch])  for i in range(input_num))

def batch_gen_with_pair(df,alphabet, batch_size=10):
    pairs=[]
    for question in df["question"].unique():
        group= df[df["question"]==question]
        pos_answers = group[df["flag"]==1]["answer"]
        neg_answers = group[df["flag"]==0]["answer"].reset_index()
        question_indices=encode_to_split(question,alphabet)
        for pos in pos_answers:
            if len(neg_answers.index)>0:
                neg_index=np.random.choice(neg_answers.index)

                neg= neg_answers.loc[neg_index,]["answer"]

                pairs.append((question_indices,encode_to_split(pos,alphabet),encode_to_split(neg,alphabet)))
    print 'pairs:{}'.format(len(pairs))
    # n_batches= int(math.ceil(df["flag"].sum()*1.0/batch_size))
    n_batches= int(len(pairs)*1.0/batch_size)
    pairs = sklearn.utils.shuffle(pairs,random_state =132)

    for i in range(0,n_batches):
        batch = pairs[i*batch_size:(i+1) * batch_size]
        yield ([pair[i] for pair in batch]  for i in range(3))

def batch_gen_with_pair_whole(df,alphabet, batch_size=10):
    pairs=[]
    for question in df["question"].unique():
        group= df[df["question"]==question]
        pos_answers = group[df["flag"]==1]["answer"]
        neg_answers = group[df["flag"]==0]["answer"]
        question_indices=encode_to_split(question,alphabet)

        for pos in pos_answers:
            for neg in neg_answers:                  
                pairs.append((question_indices,encode_to_split(pos,alphabet),encode_to_split(neg,alphabet)))
    print 'pairs:{}'.format(len(pairs))
    # n_batches= int(math.ceil(df["flag"].sum()*1.0/batch_size))
    n_batches= int(len(pairs)*1.0/batch_size)
    pairs = sklearn.utils.shuffle(pairs,random_state =132)

    for i in range(0,n_batches):
        batch = pairs[i*batch_size:(i+1) * batch_size]
        yield ([pair[i] for pair in batch]  for i in range(3))
def batch_gen_with_pair_test(df,alphabet, batch_size=10):
    pairs=[]
    for question in df["question"].unique():
        group= df[df["question"]==question]["answer"]
        question_indices=encode_to_split(question,alphabet)
        for pos in group:
            pairs.append((question_indices,encode_to_split(pos,alphabet),encode_to_split(pos,alphabet)))
    # n_batches= int(math.ceil(df["flag"].sum()*1.0/batch_size))
    n_batches= int(len(pairs)*1.0/batch_size)

    for i in range(0,n_batches-1):
        batch = pairs[i*batch_size:(i+1) * batch_size]
        yield ([pair[i] for pair in batch]  for i in range(3))


def batch_gen(X, batch_size):
    n_batches = X.shape[0]/float(batch_size)
    n_batches = int(math.ceil(n_batches))
    end = int(X.shape[0]/float(batch_size)) * batch_size
    n = 0
    for i in range(0,n_batches):
        if i < n_batches - 1:
            if len(X.shape) > 1:
                batch = X[i*batch_size:(i+1) * batch_size, :]
                yield batch
            else:
                batch = X[i*batch_size:(i+1) * batch_size]
                yield batch

        else:
            if len(X.shape) > 1:
                batch = X[end: , :]
                n += X[end:, :].shape[0]
                yield batch
            else:
                batch = X[end:]
                n += X[end:].shape[0]
                yield batch
def removeUnanswerdQuestion(df):
    counter= df.groupby("question").apply(lambda group: sum(group["flag"]))
    questions_have_correct=counter[counter>0].index
    counter= df.groupby("question").apply(lambda group: sum(group["flag"]==0))
    questions_have_uncorrect=counter[counter>0].index
    counter=df.groupby("question").apply(lambda group: len(group["flag"]))
    questions_multi=counter[counter>1].index

    return df[df["question"].isin(questions_have_correct) &  df["question"].isin(questions_have_correct) & df["question"].isin(questions_have_uncorrect)].reset_index()

def load(dataset = dataset, filter=False):

    data_dir="data/"+dataset
    train_file=os.path.join(data_dir,"train.txt")
    test_file=os.path.join(data_dir,"test.txt")
    dev_file = os.path.join(data_dir,'dev.txt')
    

    # train=pd.read_csv(train_file,header=None,sep="\t",names=["question","answer","flag"],quoting =3)
    test=pd.read_csv(test_file,header=None,sep="\t",names=["question","answer","flag"],quoting =3)
    dev = pd.read_csv(dev_file,header = None,sep = '\t',names = ['question','answer','flag'],quoting = 3)
    # if dataset == 'trec':
    #     train_all_file = os.path.join(data_dir,"train-all.txt")
    #     train_all = pd.read_csv(train_all_file,header = None,sep = '\t',names = ['qid1','qid2',
    #     'question','answer','flag'],quoting = 3)
    #     train = train_all[['question','answer','flag']]
    # else:
    train = pd.read_csv(train_file,header=None,sep="\t",names=["question","answer","flag"],quoting =3)

    if filter == True:
        return removeUnanswerdQuestion(train),removeUnanswerdQuestion(test),removeUnanswerdQuestion(dev)
    return train,test,dev

def sentence_index(sen, alphabet, input_lens):
    sen = sen.split()
    sen_index = []
    for word in sen:
        sen_index.append(alphabet[word])
    sen_index = sen_index[:input_lens]
    while len(sen_index) < input_lens:
        sen_index += sen_index[:(input_lens - len(sen_index))]

    return np.array(sen_index), len(sen)


def getSubVectors1(vectors,vocab):
    embeddings=[]
    for word in vocab:

        if word in vectors.vocab:
            embeddings.append(vectors.word_vec(word ))
        else:
            embeddings.append(np.random.uniform(-1,1,vectors.syn0.shape[1]) ) #.tolist()

    return embeddings
def getSubVectors(vectors,vocab,dim = 50):
    embedding = np.zeros((len(vocab), vectors.syn0.shape[1]))
    for word in vocab:
        if word in vectors.vocab:
            embedding[vocab[word]]= vectors.word_vec(word)
        else:
            embedding[vocab[word]]= np.random.uniform(-1,1,vectors.syn0.shape[1])  #.tolist()
    return embedding
def prepare(cropuses,is_embedding_needed = False):
    alphabet = Alphabet(start_feature_id=0)
    alphabet.add('UNKNOWN_WORD_IDX_0')
    alphabet.add('END')
    for corpus in cropuses:
        for texts in [corpus["question"],corpus["answer"]]:
            for sentence in texts:
                tokens = sentence.lower().split()
                for token in sentence.lower().split():
                    alphabet.add(token)
    if is_embedding_needed:
        fname="embedding/aquaint+wiki.txt.gz.ndim=50.bin"
        # vectors,layer1_size= load_bin_vec(fname)
        # print (len(alphabet.keys()))
        # embeddings= load_vectors(vectors,alphabet.keys(),layer1_size)
        embeddings = KeyedVectors.load_word2vec_format(fname, binary=True)
        sub_embeddings = getSubVectors(embeddings,alphabet)
        return alphabet,sub_embeddings
    else:
        return alphabet


def seq_process(df,alphabet):
    gen_seq =lambda text: " ".join([ str(alphabet[str(word)]) for word in text.split() ] )
    # gen_seq =lambda text: " ".join([ str(alphabet[str(word)]) for word in text.split() +[] *(maxlen- len(text.split())) ] )
    df["question_seq"]= df["question"].apply( gen_seq)
    df["answer_seq"]= df["answer"].apply( gen_seq)
def gen_seq_fun(text,alphabet):
    return ([ alphabet[str(word)] for word in text.lower().split() ]  +[alphabet["END"]] *(max_lenght-len(text.split())))[:max_lenght]
class Seq_gener(object):
    def __init__(self,alphabet,max_lenght):
        self.alphabet = alphabet
        self.max_lenght = max_lenght
    def __call__(self, text):
        return ([ self.alphabet[str(word)] for word in text.lower().split() ]  +[self.alphabet["END"]] *(self.max_lenght-len(text.split())))[:self.max_lenght]

def getQAIndiceofTest(df,alphabet,max_lenght=50):
    gen_seq =lambda text: ([ alphabet[str(word)] for word in text.lower().split() ]  +[alphabet["END"]] *(max_lenght-len(text.split())))[:max_lenght]
    # gen_seq =lambda text: " ".join([ str(alphabet[str(word)]) for word in text.split() +[] *(maxlen- len(text.split())) ] )
    # questions= np.array(map(gen_seq,df["question"]))
    # answers= np.array(map( gen_seq,df["answer"]))
    pool = multiprocessing.Pool(cores)
    questions = pool.map(Seq_gener(alphabet,max_lenght),df["question"])
    answers = pool.map(Seq_gener(alphabet,max_lenght),df["answer"])
    return [np.array(questions),np.array(answers)]
def getDataFolexdecomp():
    train,test,dev = load("trec",filter=True)
    alphabet,embeddings = prepare([train,test,dev],is_embedding_needed = True)
    print 'load finished'
    voc = open('GoogleNews-vectors-50d.voc','w')
    for key in alphabet:
        voc.write(key+'\n')
    np.save('GoogleNews-vectors-50d',embeddings)
    print 'save finished'
# load data for trec sigar 2015

def get_alphabet(corpuses):
    alphabet = Alphabet(start_feature_id=0)
    alphabet.add('UNKNOWN_WORD_IDX')
    for corpus in corpuses:
        for texts in [corpus["question"],corpus["answer"]]:
            for sentence in texts:
                tokens = sentence.lower().split()
                for token in sentence.lower().split():
                    alphabet.add(token)
    return alphabet
    print len(alphabet)  
def compute_df(corpus):
    word2df = defaultdict(float)
    numdoc = len(corpus['question']) + len(corpus['answer'])
    for texts in [corpus['question'].unique(),corpus['answer']]:
        for sentence in texts:
            tokens = sentence.lower().split()
            for w in set(tokens):
                word2df[w] += 1.0;


    for w,value in word2df.iteritems():
        word2df[w] /= np.math.log(numdoc / value) 
    return word2df


#get onerlap
def compute_overlap_features(dataset,stoplist = None,word2df = None):

    word2df = word2df if word2df else {}
    stoplist = stoplist if stoplist else set()
    question = dataset['question'].lower().split()
    answer = dataset['answer'].lower().split()
    q_set = set([q for q in question if q not in stoplist])
    a_set = set([a for a in answer if a not in stoplist])
    word_overlap = q_set.intersection(a_set)
    overlap = float(len(word_overlap)) / (len(q_set) + len(a_set))
    word_overlap = q_set.intersection(a_set)
    df_overlap = 0.0
    for w in word_overlap:
        df_overlap += word2df[w]
    df_overlap /= (len(q_set) + len(a_set))

    feats_overlap = np.array([overlap,df_overlap])
    return [overlap,df_overlap] * 2
def compute_overlap_idx(dataset, stoplist, q_max_sent_length, a_max_sent_length):
    stoplist = stoplist if stoplist else set()
    questions = dataset['question'].str.split()
    answers = dataset['answer'].str.split()
    q_indices, a_indices = [], []
    for question,answer in zip(questions,answers):    
        q_set = [q.lower() for q in question if q not in stoplist]
        a_set = [a.lower() for a in answer if a not in stoplist]
        word_overlap = set(q_set).intersection(set(a_set))

        # question index
        q_idx = np.ones(q_max_sent_length) * 2
        for i,q in enumerate(question):
            value = 0
            if q in word_overlap:
                value = 1
            q_idx[i] = value

        q_indices.append(q_idx)
        a_idx = np.ones(a_max_sent_length) * 2
        # answer index
        for i,a in enumerate(answer):
            value = 0
            if a in word_overlap:
                value = 1
            a_idx[i] = value
        a_indices.append(a_idx)
    q_indices = np.vstack(q_indices).astype('int32')
    a_indices = np.vstack(a_indices).astype('int32')
    return q_indices,a_indices
def convert2indices(dataset,alphabet,dummy_word_idx,max_sent_length=40):
    data_idx = []
    sentences = dataset.str.split()
    for sentence in sentences:
        ex = np.ones(max_sent_length) * dummy_word_idx
        for i,token in enumerate(sentence):
            idx = alphabet.get(token.lower(), UNKNOWN_WORD_IDX)
            ex[i] = idx
        data_idx.append(ex)
    data_idx = np.asarray(data_idx).astype('int32')
    return data_idx

def loadData(dataset = dataset):
    data_dir = "data/"+dataset
    train_file = os.path.join(data_dir,"train.txt")
    test_file = os.path.join(data_dir,'test.txt')
    dev_file = os.path.join(data_dir,'dev.txt')
    train = pd.read_csv(train_file,header=None,sep="\t",names=["question","answer","flag"],quoting =3)
    test = pd.read_csv(test_file,header = None,sep = '\t',names = ['question','answer','flag'],quoting = 3)
    dev = pd.read_csv(dev_file,header = None,sep = '\t',names = ['question','answer','flag'],quoting = 3)

    alphabet = get_alphabet([train,test,dev])

    cPickle.dump(alphabet, open(os.path.join(data_dir, 'vocab.pickle'), 'w'))
    all_file = pd.concat([train,test,dev])
    q_max_sent_length = max(map(lambda x:len(x),all_file['question'].str.split()))
    a_max_sent_length = max(map(lambda x:len(x),all_file['answer'].str.split()))
    word2dfs = compute_df(all_file)

    print 'q_max_sent_length', q_max_sent_length
    print 'a_max_sent_length', a_max_sent_length
    print 'alphabet',len(alphabet)
    

    for fname in [train_file,dev_file,test_file]:
        dataset = pd.read_csv(fname,header = None,sep = '\t',names = ['question','answer','flag'],quoting = 3)
        dataset['overlapfeats'] = dataset.apply(compute_overlap_features,word2df = word2dfs, axis = 1)  
        question2id = dict()
        for index, question in enumerate(dataset['question'].unique()):
            question2id[question] = index
        dataset['qid'] = dataset.apply(lambda row:question2id[row['question']],axis = 1)

        # dataset['qa_overlap_indices'] = dataset.apply(compute_overlap_idx,stoplist = None,
        #     q_max_sent_length = q_max_sent_length,a_max_sent_length = a_max_sent_length,axis = 1)
        q_overlap_indices,a_overlap_indices = compute_overlap_idx(dataset,stoplist = None,q_max_sent_length = q_max_sent_length,
            a_max_sent_length = a_max_sent_length)
        questions_idx = convert2indices(dataset['question'],alphabet = alphabet,
            dummy_word_idx = alphabet.fid,max_sent_length = q_max_sent_length)
        answers_idx = convert2indices(dataset['answer'],alphabet = alphabet,
            dummy_word_idx = alphabet.fid,max_sent_length = a_max_sent_length)
        print 'answers_idx', answers_idx.shape

        qids = np.array(dataset['qid']).astype('int32')
        flags = np.array(dataset['flag']).astype('int32')
        overlapfeats = np.array(dataset['overlapfeats'])
        overlapfeats = np.vstack(overlapfeats).astype('int32')

        basename,_ = os.path.splitext(os.path.basename(fname))

        print 'qids shape',qids.shape
        print 'flags shape',flags.shape
        print 'question shape',questions_idx.shape
        print 'answer shape',answers_idx.shape
        print 'overlap_feats shape',overlapfeats.shape

        print 'q_overlap_indices shape',q_overlap_indices.shape
        print 'a_overlap_indices shape',a_overlap_indices.shape

        np.save(os.path.join(data_dir, '{}.qids.npy'.format(basename)), qids)
        np.save(os.path.join(data_dir, '{}.questions.npy'.format(basename)), questions_idx)
        np.save(os.path.join(data_dir, '{}.answers.npy'.format(basename)), answers_idx)
        np.save(os.path.join(data_dir, '{}.labels.npy'.format(basename)), flags)
        np.save(os.path.join(data_dir, '{}.overlap_feats.npy'.format(basename)), overlapfeats)

        np.save(os.path.join(data_dir, '{}.q_overlap_indices.npy'.format(basename)), q_overlap_indices)
        np.save(os.path.join(data_dir, '{}.a_overlap_indices.npy'.format(basename)), a_overlap_indices)


        # print 'questions',len(dataset['qid'].unique())

        # questions_idx = np.array(dataset['questions_answers_idx']).shape
   
        
       
        
  

    # question2id = dict()
    # for index,q in enumerate(all_file['question'].unique()):
    #     question2id[q] = index
    # all_file['qids'] = all_file.apply(lambda row:question2id[row['question']],axis = 1)


def data_processing():
    train,test,dev = load('trec',filter = False)
    q_max_sent_length = max(map(lambda x:len(x),train['question'].str.split()))
    a_max_sent_length = max(map(lambda x:len(x),train['answer'].str.split()))
    print('Total number of unique question:{}'.format(len(train['question'].unique())))
    print('Total number of question pairs for training: {}'.format(len(train)))
    print('Total number of question pairs for test: {}'.format(len(test)))
    print('Total number of question pairs for dev: {}'.format(len(dev)))
    print('Duplicate pairs: {}%'.format(round(train['flag'].mean()*100, 2)))
    print(len(train['question'].unique()))

    #text analysis
    train_qs = pd.Series(train['answer'].tolist())
    test_qs = pd.Series(test['answer'].tolist())

    dist_train = train_qs.apply(lambda x:len(x.split(' ')))
    dist_test = test_qs.apply(lambda x:len(x.split(' ')))
    pal = sns.color_palette()
    plt.figure(figsize=(15, 10))
    plt.hist(dist_train, bins=200, range=[0, 200], color=pal[2], normed=True, label='train')
    plt.hist(dist_test, bins=200, range=[0, 200], color=pal[1], normed=True, alpha=0.5, label='test')
    plt.title('Normalised histogram of character count in questions', fontsize=15)
    plt.legend()
    plt.xlabel('Number of words', fontsize=15)
    plt.ylabel('Probability', fontsize=15)

    print('mean-train {:.2f} std-train {:.2f} mean-test {:.2f} std-test {:.2f} max-train {:.2f} max-test {:.2f}'.format(dist_train.mean(), 
                          dist_train.std(), dist_test.mean(), dist_test.std(), dist_train.max(), dist_test.max()))
    plt.show('hard')

    qmarks = np.mean(train_qs.apply(lambda x: '?' in x))
    who = np.mean(train_qs.apply(lambda x:'Who' in x))
    where = np.mean(train_qs.apply(lambda x:'Where' in x))
    how_many = np.mean(train_qs.apply(lambda x:'How many' in x))
    fullstop = np.mean(train_qs.apply(lambda x: '.' in x))
    capital_first = np.mean(train_qs.apply(lambda x: x[0].isupper()))
    capitals = np.mean(train_qs.apply(lambda x: max([y.isupper() for y in x])))
    numbers = np.mean(train_qs.apply(lambda x: max([y.isdigit() for y in x])))
    print('Questions with question marks: {:.2f}%'.format(qmarks * 100))
    print('Questions with [Who] tags: {:.2f}%'.format(who * 100))
    print('Questions with [where] tags: {:.2f}%'.format(where * 100))
    print('Questions with [How many] tags:{:.2f}%'.format(how_many * 100))
    print('Questions with full stops: {:.2f}%'.format(fullstop * 100))
    print('Questions with capitalised first letters: {:.2f}%'.format(capital_first * 100))
    print('Questions with capital letters: {:.2f}%'.format(capitals * 100))
    print('Questions with numbers: {:.2f}%'.format(numbers * 100))

def main():
    train,test,dev = load("trec",filter=False)
    alphabet,embeddings = prepare([train,test,dev],is_embedding_needed = True)
    # seq_process(train, alphabet)
    # seq_process(test, alphabet)
    x,y = getQAIndiceofTest(test,alphabet)
    print (x)
    print (type(x))
    print (x.shape)
    # for batch in batch_gen_with_single(train,alphabet,10):

    #     x,y=batch
    #     print (len(x))
        # exit()
def sampleTest():
    train_file = os.path.join('data/trec',"train.txt")
    train = pd.read_csv(train_file,header=None,sep="\t",names=["question","answer","flag"],quoting =3)
    alphabet = get_alphabet([train])
    alphabet.add('END')
    for question,answer,answer_negtive in batch_gen_with_pair(train,alphabet):
        print question,answer,answer_negtive
def random_result():
    train,test,dev = load("wiki",filter = True)
    test = test.reindex(np.random.permutation(test.index))

    # test['pred'] = test.apply(idf_word_overlap,axis = 1)
    pred = np.random.randn(len(test))

    print evaluation.evaluationBypandas(test,pred)
if __name__ == '__main__':
    # sampleTest()
    # data_processing()
    random_result()
    # alphabet = prepare([train,test,dev],is_embedding_needed = False)
    # embedding_file = 'embedding/GoogleNews-vectors-negative300.bin'
    # load_bin_vec(embedding_file,alphabet.keys())
    # getDataFolexdecomp()
    # main()

