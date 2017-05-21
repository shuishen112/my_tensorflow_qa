# -*- coding:utf-8-*-
import numpy as np
import random,os,math
import pandas as pd
from gensim.models.keyedvectors import KeyedVectors
import sklearn
import multiprocessing
import time
import cPickle as pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import evaluation
from features import overlap_jiabing,overlap_index
import string
import jieba
from nltk import stem
import chardet
PUNCT = set(string.punctuation) - set('$%#')
print PUNCT
cores = multiprocessing.cpu_count()
dataset= "wiki"
UNKNOWN_WORD_IDX = 0
is_stemmed_needed = False



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


isEnglish = False
if is_stemmed_needed:
    stemmer = stem.lancaster.LancasterStemmer()
class Alphabet(dict):
    def __init__(self, start_feature_id = 1):
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

# get data for lexdecomp
@log_time_delta
def load_bin_vec(words,fname='embedding/GoogleNews-vectors-negative300.bin'):
    print fname
    vocab = set(words)
    word_vecs = {}
    # voc = open('GoogleNews-vectors-300d.voc','w')
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
            if is_stemmed_needed:
                temp_word = stemmer.stem(word.decode('utf-8').lower())
            else:
                temp_word = word.decode('utf-8').lower()
            if temp_word in vocab:
                # the vector fo word is just like this:'\x01\x02'
                word_vecs[temp_word] = np.fromstring(f.read(binary_len),dtype = 'float32')
                # voc.write(temp_word + '\n')
                embedding.append(word_vecs[temp_word])
            else:
                # print word
                f.read(binary_len)
        print 'done'
        print 'words found in wor2vec embedding ',len(word_vecs.keys())
        # np.save('GoogleNews-vectors-300d',embedding)
        return word_vecs
def load_text_vec(alphabet,filename="",embedding_size = 100):
    vectors = {}
    with open(filename) as f:
        i=0
        for line in f:
            i+=1
            if i % 100000 == 0:
                    print 'epch %d' % i
            items = line.strip().split(' ')
            if len(items) == 2:
                vocab_size, embedding_size= items[0],items[1]
                print ( vocab_size, embedding_size)
            else:
                word = items[0]
                if word in alphabet:
                    vectors[word] = items[1:]
    print 'embedding_size',embedding_size
    print 'done'
    print 'words found in wor2vec embedding ',len(vectors.keys())
    return vectors
def load_text_vector_test(filename = "",embedding_size = 100):
    vectors = {}
    with open(filename) as f:
        i=0
        for line in f:
            i+=1
            if i % 100000 == 0:
                    print 'epch %d' % i
            items = line.strip().split(' ')
            if len(items) == 2:
                vocab_size, embedding_size= items[0],items[1]
                print ( vocab_size, embedding_size)
            else:
                word = items[0]
                print word
                if i > 1000:
                    exit()
                vectors[word] = items[1:]
    print 'embedding_size',embedding_size
    print 'done'
    print 'words found in wor2vec embedding ',len(vectors.keys())
    return vectors
def load_vectors( vectors,vocab,dim_size):
    if vocab==None:
        return
    embeddings=[]
    for word in vocab:
        embeddings.append(vectors.get(word,np.random.uniform(-1,1,dim_size).tolist() ))
    return embeddings
def encode_to_split(sentence,alphabet,max_sentence = 40):
    indices = []
    if is_stemmed_needed:
        tokens = [stemmer.stem(w.decode('utf-8')) for w in sentence.strip().lower().split() if w not in PUNCT]
    else:
        # tokens = [w for w in sentence.strip().lower().split() if w not in PUNCT]
        tokens = cut(sentence)
    for word in tokens:
        indices.append(alphabet[word])
    results=indices+[alphabet["END"]]*(max_sentence-len(indices))
    return results[:max_sentence]
def transform(flag):
    if flag == 1:
        return [0,1]
    else:
        return [1,0]
@log_time_delta
def batch_gen_with_single(df,alphabet,batch_size = 10,q_len = 33,a_len = 40):
    pairs=[]
    for index,row in df.iterrows():
        quetion = encode_to_split(row["question"],alphabet,max_sentence = q_len)
        answer = encode_to_split(row["answer"],alphabet,max_sentence = a_len)
        q_pos_overlap,a_pos_overlap = overlap_index(row["question"],row["answer"],q_len,a_len)
        pairs.append((quetion,answer,q_pos_overlap,a_pos_overlap))
    # n_batches= int(math.ceil(df["flag"].sum()*1.0/batch_size))
    n_batches = int(len(pairs)*1.0/batch_size)
    # pairs = sklearn.utils.shuffle(pairs,random_state =132)
    for i in range(0,n_batches):
        batch = pairs[i*batch_size:(i+1) * batch_size]

        yield [[pair[j] for pair in batch]  for j in range(4)]
    batch= pairs[n_batches*batch_size:] + [pairs[n_batches*batch_size]] * (batch_size- len(pairs)+n_batches*batch_size  )
    yield [[pair[i] for pair in batch]  for i in range(4)]
@log_time_delta
def batch_gen_with_single_attentive(df,alphabet,batch_size = 10,q_len = 33,a_len = 40):
    pairs=[]
    for index,row in df.iterrows():
        quetion = encode_to_split(row["question"],alphabet,max_sentence = q_len)
        answer = encode_to_split(row["answer"],alphabet,max_sentence = a_len)
        pairs.append((quetion,answer))
    # n_batches= int(math.ceil(df["flag"].sum()*1.0/batch_size))
    n_batches = int(len(pairs)*1.0/batch_size)
    # pairs = sklearn.utils.shuffle(pairs,random_state =132)
    for i in range(0,n_batches):
        batch = pairs[i*batch_size:(i+1) * batch_size]

        yield [[pair[j] for pair in batch]  for j in range(2)]
    batch= pairs[n_batches*batch_size:] + [pairs[n_batches*batch_size]] * (batch_size- len(pairs)+n_batches*batch_size  )
    yield [[pair[i] for pair in batch]  for i in range(2)]
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
@log_time_delta
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
@log_time_delta
def batch_gen_with_pair(df,alphabet, batch_size=10,q_len = 40,a_len = 40):
    pairs=[]
    for question in df["question"].unique():
        group= df[df["question"]==question]
        pos_answers = group[df["flag"]==1]["answer"]
        neg_answers = group[df["flag"]==0]["answer"].reset_index()
        question_indices = encode_to_split(question,alphabet,max_sentence = q_len)
        for pos in pos_answers:
            if len(neg_answers.index)>0:
                neg_index=np.random.choice(neg_answers.index)

                neg= neg_answers.loc[neg_index,]["answer"]

                pairs.append((question_indices,encode_to_split(pos,alphabet,max_sentence = a_len),encode_to_split(neg,alphabet,max_sentence = a_len)))
    print 'pairs:{}'.format(len(pairs))
    # n_batches= int(math.ceil(df["flag"].sum()*1.0/batch_size))
    n_batches= int(len(pairs)*1.0/batch_size)
    pairs = sklearn.utils.shuffle(pairs,random_state =132)

    for i in range(0,n_batches):
        batch = pairs[i*batch_size:(i+1) * batch_size]
        yield [[pair[i] for pair in batch]  for i in range(3)]
def overlap_index(question,answer,q_len,a_len,stopwords = []):
    qset = set(cut(question))
    aset = set(cut(answer))

    q_index = np.zeros(q_len)
    a_index = np.zeros(a_len)

    overlap = qset.intersection(aset)
    for i,q in enumerate(cut(question)[:q_len]):
        value = 1
        if q in overlap:
            value = 2
        q_index[i] = value
    for i,a in enumerate(cut(answer)[:a_len]):
        value = 1
        if a in overlap:
            value = 2
        a_index[i] = value
    return q_index,a_index
def batch_gen(df,alphabet, batch_size = 10,q_len = 40,a_len = 40):
    pairs=[]
    for question in df["question"].unique():
        group= df[df["question"]==question]
        pos_answers = group[df["flag"]==1]["answer"]
        neg_answers = group[df["flag"]==0]["answer"].reset_index()
        question_indices = encode_to_split(question,alphabet,max_sentence = q_len)
        for pos in pos_answers:
            if len(neg_answers.index)>0:
                neg_index=np.random.choice(neg_answers.index)
@log_time_delta
def batch_gen_with_pair_overlap(df,alphabet, batch_size = 10,q_len = 40,a_len = 40):
    pairs=[]
    for question in df["question"].unique():
        group= df[df["question"]==question]
        pos_answers = group[df["flag"]==1]["answer"]
        neg_answers = group[df["flag"]==0]["answer"].reset_index()
        question_indices = encode_to_split(question,alphabet,max_sentence = q_len)
        for pos in pos_answers:
            if len(neg_answers.index)>0:
                neg_index=np.random.choice(neg_answers.index)

                neg = neg_answers.loc[neg_index,]["answer"]
                q_pos_overlap,a_pos_overlap=overlap_index(question,pos,q_len,a_len)
                
                q_neg_overlap,a_neg_overlap=overlap_index(question,neg,q_len,a_len)
                pairs.append((question_indices,encode_to_split(pos,alphabet,max_sentence = a_len),encode_to_split(neg,alphabet,max_sentence = a_len),q_pos_overlap,q_neg_overlap,a_pos_overlap,a_neg_overlap))
    print 'pairs:{}'.format(len(pairs))
    # n_batches= int(math.ceil(df["flag"].sum()*1.0/batch_size))
    n_batches= int(len(pairs)*1.0/batch_size)
    pairs = sklearn.utils.shuffle(pairs,random_state =132)

    for i in range(0,n_batches):
        batch = pairs[i*batch_size:(i+1) * batch_size]
        yield [[pair[i] for pair in batch]  for i in range(7)]
def get_overlap_dict(df,alphabet,q_len = 40,a_len = 40):
    d = dict()
    for question in df['question'].unique():
        group = df[df['question'] == question]
        pos_answers = group[df["flag"]==1]["answer"]
        neg_answers = group[df["flag"]==0]["answer"].reset_index()
        for pos in pos_answers:
            if len(neg_answers.index)>0:
                neg_index = np.random.choice(neg_answers.index)

                neg = neg_answers.loc[neg_index,]["answer"]
                q_pos_overlap,a_pos_overlap = overlap_index(question,pos,q_len,a_len)
                d[(question,pos)] = (q_pos_overlap,a_pos_overlap)
                q_neg_overlap,a_neg_overlap = overlap_index(question,neg,q_len,a_len)
                d[(question,neg)] = 


def get_raw_pairs(df):
    pairs = []
    for question in df['question'].unique():
        group = df[df['question'] == question]
        pos_answers = group[df['flag'] == 1]['answer']
        neg_answers = group[df['flag'] == 0]['answer'].reset_index()
        for pos in pos_answers:
            if len(neg_answers.index) > 0:
                neg_index = np.random.choice(neg_answers.index)
                neg = neg_answers.loc[neg_index]['answer']
                pairs.append((question,pos,neg))
    return pairs
def batch_gen_with_overlap(df,alphabet,batch_size = 10,qlen = 40,a_len = 40):
    pairs = get_raw_pairs(df)
    overlap = []
    for question,pos,neg in pairs:
        q_pos_overlap_index,a_pos_overlap_index = overlap_index(question,pos,q_len,a_len)
        q_neg_overlap_index,a_neg_overlap_index = overlap_index(question,neg,q_len,a_len)
def batch_gen_with_pare_output_feature(df,alphabet,batch_size = 10,q_len = 40,a_len = 40):
    pairs = get_raw_pairs(df)

    overlap = []
    for question,pos,neg in pairs:
        q_pos_overlap_index,a_pos_overlap_index = overlap_index(question,pos,q_len,a_len)
        q_neg_overlap_index,a_neg_overlap_index = overlap_index(question,neg,q_len,a_len)    # n_batches= int(math.ceil(df["flag"].sum()*1.0/batch_size))
    n_batches= int(len(pairs)*1.0/batch_size)
    pairs = sklearn.utils.shuffle(pairs,random_state =132)

    for i in range(0,n_batches):
        batch = pairs[i*batch_size:(i+1) * batch_size]
        yield ([pair[i] for pair in batch]  for i in range(3))
def batch_gen_with_pair_whole(df,alphabet, batch_size = 10,q_len = 40,a_len = 40):
    pairs=[]
    for question in df["question"].unique():
        group= df[df["question"]==question]
        pos_answers = group[df["flag"]==1]["answer"]
        neg_answers = group[df["flag"]==0]["answer"]
        question_indices=encode_to_split(question,alphabet,max_sentence = q_len)

        for pos in pos_answers:
            for neg in neg_answers:                  
                pairs.append((question_indices,encode_to_split(pos,alphabet,max_sentence = a_len),encode_to_split(neg,alphabet,max_sentence = a_len)))
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
    if dataset == 'trec':
        train_all_file = os.path.join(data_dir,"train-all.txt")
        train_all = pd.read_csv(train_all_file,header = None,sep = '\t',names = ['qid1','qid2',
        'question','answer','flag'],quoting = 3)
        train = train_all[['question','answer','flag']]
    else:
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


def getSubVectorsFromDict(vectors,vocab,dim = 300):
    embedding = np.zeros((len(vocab),dim))
    for word in vocab:
        if word in vectors:
            embedding[vocab[word]]= vectors[word]
        else:
            embedding[vocab[word]]= np.random.uniform(-0.25,0.25,dim) #.tolist()

    return embedding
def getSubVectors(vectors,vocab,dim = 50):
    print 'embedding_size:',vectors.syn0.shape[1]
    embedding = np.zeros((len(vocab), vectors.syn0.shape[1]))
    for word in vocab:
        if word in vectors.vocab:
            embedding[vocab[word]]= vectors.word_vec(word)
        else:
            embedding[vocab[word]]= np.random.uniform(-0.5,+0.5,vectors.syn0.shape[1])  #.tolist()
    return embedding
def cut(sentence,isEnglish = isEnglish):
    if isEnglish:
        tokens = sentence.lower().split()
    else:
        stopwords = { word.decode("utf-8") for word in open("model/chStopWordsSimple.txt").read().split()}

        # words = jieba.cut(str(sentence))
        tokens = [word for word in sentence.split() if word not in stopwords]
    return tokens
@log_time_delta
def prepare(cropuses,is_embedding_needed = False,dim = 50,fresh = False):
    vocab_file = 'model/voc'
    if os.path.exists(vocab_file) and not fresh:
        alphabet = pickle.load(open(vocab_file,'r'))
    else:   
        alphabet = Alphabet(start_feature_id=0)
        alphabet.add('UNKNOWN_WORD_IDX_0')  
        alphabet.add('END') 
        count = 0
        for corpus in cropuses:
            for texts in [corpus["question"].unique(),corpus["answer"]]:
                for sentence in texts:   
                    count += 1
                    if count % 10000 == 0:
                        print count
                    tokens = cut(sentence)
                    # print "#".join(tokens)
                    for token in tokens:
                        if is_stemmed_needed:
                            # try:
                            alphabet.add(stemmer.stem(token.decode('utf-8')))
                            # except Exception as e:
                            #     alphabet.add(token)
                            #     print type(e)
                            
                        else:

                            alphabet.add(token)
        print 'count sentence',count
        pickle.dump(alphabet,open(vocab_file,'w'))
    if is_embedding_needed:
        sub_vec_file = 'embedding/sub_vector'
        if os.path.exists(sub_vec_file) and not fresh:
            sub_embeddings = pickle.load(open(sub_vec_file,'r'))
        else:    
            if isEnglish:        
                if dim == 50:
                    fname = "embedding/aquaint+wiki.txt.gz.ndim=50.bin"
                    embeddings = KeyedVectors.load_word2vec_format(fname, binary=True)
                    sub_embeddings = getSubVectors(embeddings,alphabet)
                else:
                    fname = 'embedding/glove.6B/glove.6B.300d.txt'
                    embeddings = load_text_vec(alphabet,fname,embedding_size = dim)
                    sub_embeddings = getSubVectorsFromDict(embeddings,alphabet,dim)
            else:
                fname = 'model/dbqa.word2vec'
                embeddings = load_text_vec(alphabet,fname,embedding_size = dim)
                sub_embeddings = getSubVectorsFromDict(embeddings,alphabet,dim)
            pickle.dump(sub_embeddings,open(sub_vec_file,'w'))
        # print (len(alphabet.keys()))
        # embeddings = load_vectors(vectors,alphabet.keys(),layer1_size)
        # embeddings = KeyedVectors.load_word2vec_format(fname, binary=True)
        # sub_embeddings = getSubVectors(embeddings,alphabet)
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
    alphabet = Alphabet(start_feature_id = 0)
    alphabet.add('UNKNOWN_WORD_IDX')
    for corpus in corpuses:
        for texts in [corpus["question"],corpus["answer"]]:
            for sentence in texts:
                tokens = sentence.lower().split()
                for token in sentence.lower().split():
                    if is_stemmed_needed:
                        alphabet.add(stemmer.stem(token.decode('utf-8')))
                    else:
                        alphabet.add(token)
    print len(alphabet)  
    return alphabet
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
        print questions_idx
        exit()
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
    train,test,dev = load('nlpcc',filter = True)
    q_max_sent_length = max(map(lambda x:len(x),train['question'].str.split()))
    a_max_sent_length = max(map(lambda x:len(x),train['answer'].str.split()))
    q_len = map(lambda x:len(x),train['question'].str.split())
    a_len = map(lambda x:len(x),train['answer'].str.split())
    print np.max(q_len)
    print np.max(a_len)
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
def overlap_visualize():
    train,test,dev = load("nlpcc",filter=True)
    test = test.reindex(np.random.permutation(test.index))
    df = test
    df['qlen'] = df['question'].str.len()
    df['alen'] = df['answer'].str.len()

    df['q_n_words'] = df['question'].apply(lambda row:len(row.split(' ')))
    df['a_n_words'] = df['answer'].apply(lambda row:len(row.split(' ')))

    def normalized_word_share(row):
        w1 = set(map(lambda word: word.lower().strip(), row['question'].split(" ")))
        w2 = set(map(lambda word: word.lower().strip(), row['answer'].split(" ")))    
        return 1.0 * len(w1 & w2)/(len(w1) + len(w2))
    df['word_share'] = df.apply(normalized_word_share, axis=1)

    plt.figure(figsize=(12, 8))
    plt.subplot(1,2,1)
    sns.violinplot(x = 'flag', y = 'word_share', data = df[0:50000])
    plt.subplot(1,2,2)
    sns.distplot(df[df['flag'] == 1.0]['word_share'][0:10000], color = 'green')
    sns.distplot(df[df['flag'] == 0.0]['word_share'][0:10000], color = 'red')

    print evaluation.evaluationBypandas(test,df['word_share'])
    plt.show('hold')
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
def random_result():
    train,test,dev = load("wiki",filter = True)
    # test = test.reindex(np.random.permutation(test.index))

    # test['pred'] = test.apply(idf_word_overlap,axis = 1)
    pred = np.random.randn(len(test))

    print evaluation.evaluationBypandas(test,pred)
def dns_sample(df,alphabet,q_len,a_len,sess,model,batch_size,neg_sample_num = 10):
    samples = []
    count = 0
    # neg_answers = df['answer'].reset_index()
    pool_answers = df[df.flag==1]['answer'].tolist()
    # pool_answers = df[df['flag'] == 0]['answer'].tolist()
    print 'question unique:{}'.format(len(df['question'].unique()))
    for question in df['question'].unique():
        group = df[df['question'] == question]
        pos_answers = group[df["flag"]==1]["answer"].tolist()
        pos_answers_exclude = list(set(pool_answers).difference(set(pos_answers)))
        neg_answers = group[df["flag"]==0]["answer"].tolist()
        question_indices = encode_to_split(question,alphabet,max_sentence = q_len)
        for pos in pos_answers:
            # negtive sample
            neg_pool = []
            if len(neg_answers) > 0:
   
                neg_exc = list(np.random.choice(pos_answers_exclude,size = 100 - len(neg_answers)))
                neg_answers_sample = neg_answers + neg_exc
                # neg_answers = neg_a
                # print 'neg_tive answer:{}'.format(len(neg_answers))
                for neg in neg_answers_sample:
                    neg_pool.append(encode_to_split(neg,alphabet,max_sentence = a_len))
                # for i in range(neg_sample_num):
                #     # neg_index = np.random.choice(neg_answers.index)
                #     # neg = neg_answers.loc[neg_index]["answer"]
                #     neg = np.random.choice(neg_answers)
                #     neg_pool.append((question_indices,encode_to_split(pos,alphabet,max_sentence = a_len),encode_to_split(neg,alphabet,max_sentence = a_len)))
                # for i in range(30):
                #     # neg_index = np.random.choice(neg_answers.index)
                #     # neg = neg_answers.loc[neg_index]["answer"]
                #     neg = np.random.choice(pos_answers_exclude)
                #     neg_pool.append((question_indices,encode_to_split(pos,alphabet,max_sentence = a_len),encode_to_split(neg,alphabet,max_sentence = a_len)))
                # use the model to predict
                # neg_pool = np.array(neg_pool)
                # input_x_1 = list(neg_pool[:,0])
                # input_x_2 = list(neg_pool[:,1])
                # input_x_3 = list(neg_pool[:,2])
                input_x_1 = [question_indices] * len(neg_answers_sample)
                input_x_2 = [encode_to_split(pos,alphabet,max_sentence = a_len)] * len(neg_answers_sample)
                input_x_3 = neg_pool
                feed_dict = {
                    model.question: input_x_1,
                    model.answer: input_x_2,
                    model.answer_negative:input_x_3 
                }
                predicted = sess.run(model.score13,feed_dict)
                # find the max score
                index = np.argmax(predicted)
                # print len(neg_answers)
                # print 'index:{}'.format(index)
                # if len(neg_answers)>1:
                #     print neg_answers[1]
                samples.append((question_indices,encode_to_split(pos,alphabet,max_sentence = a_len),input_x_3[index]))      
                count += 1
                if count % 100 == 0:
                    print 'samples load:{}'.format(count)
    print 'samples finishted len samples:{}'.format(len(samples))
    return samples
@log_time_delta
def batch_gen_with_pair_dns(samples,batch_size,epoches=1):
    # n_batches= int(math.ceil(df["flag"].sum()*1.0/batch_size))
    n_batches = int(len(samples) * 1.0 / batch_size)
    for j in range(epoches):
        pairs = sklearn.utils.shuffle(samples,random_state =132)
        for i in range(0,n_batches):
            batch = pairs[i*batch_size:(i+1) * batch_size]
            yield ([pair[i] for pair in batch]  for i in range(3))   
def data_sample_for_dev(dataset):
    data_dir = "data/" + dataset
    train_file = os.path.join(data_dir,"train.txt")
    dev_file = os.path.join(data_dir,'dev.txt')
    train = pd.read_csv(train_file,header=None,sep="\t",names=["question","answer","flag"],quoting =3)

    dev = train.sample(frac = 0.7)
    print dev
    dev.to_csv(dev_file,index = None,sep = '\t',header = None,quoting = 3)
def sample_data(df,frac = 0.5):
    df = df.sample(frac = frac)
    df = df.reset_index(drop = True)
    return df
def replace_number(data):
    for df in data:
        df['question'] = df['question'].str.replace(r'[A-Za-z]+','[EN]')
        df['question'] = df['question'].str.replace(r'[\d]+','[NUM]')
        df['answer'] = df['answer'].str.replace(r'[A-Za-z]+','[EN]')
        df['answer'] = df['answer'].str.replace(r'[\d]+','[NUM]')
if __name__ == '__main__':
    # data_processing()
    train,test,dev = load('nlpcc',filter = True)
    train[train['flag'] == 1].to_csv('flag1.csv',index = False)
    # replace_number([train,test,dev])
    # print train
    # exit()
    # # alphabet,embeddings = prepare([train,test,dev],dim = 300,is_embedding_needed = True,fresh = True)
    # print len(alphabet)
    # file = open('word_wiki.txt','w')
    # for w in alphabet:
    #     file.write(w + '\n')
    # data_processing()
    # q_max_sent_length = max(map(lambda x:len(x),train['question'].str.split()))
    # a_max_sent_length = max(map(lambda x:len(x),train['answer'].str.split()))
    # print 'q_question_length:{} a_question_length:{}'.format(q_max_sent_length,a_max_sent_length)
    # print 'train question unique:{}'.format(len(train['question'].unique()))
    # print 'train length',len(train)
    # print 'test length', len(test)
    # print 'dev length', len(dev)
    # overlap_visualize()
    # print 'alphabet:',len(alphabet)
    # vec = load_text_vector_test(filename = "embedding/glove.6B/glove.6B.300d.txt",embedding_size = 300)
    # for k in vec.keys():
    #     print k
    # load_bin_vec(alphabet)
        # exit()
    # word = 'interesting'
    # print stemmer.stem(word)
    # train,test,dev = load("wiki",filter = True)
    # alphabet,embeddings = prepare([train,test,dev])


