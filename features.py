import cPickle as pickle
import evaluation
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk import stem
from nltk.tokenize import word_tokenize
idf_outer = pickle.load(open('data/small_idf_dict','r'))
english_punctuations = [',','.',':','?','(',')','[',']','!','@','#','%','&']
stemmer = stem.lancaster.LancasterStemmer()
# idf_outer = pickle.load(open('data/idf.pkl'))
def overlap_jiabing(row,stopwords=[]): #stopwords=nltk.corpus.stopwords.words('english')
	question = row["question"].split()
	answer = row["answer"].split()
	qindex = [q for q in question if q not in stopwords]
	# print len(qindex)
	aindex = [a for a in answer if a not in stopwords]
	qset = set(qindex)
	# overlap = [a for a in aindex if a in qset]
	aset = set(aindex)
	# overlap = [q for q in qindex if q in aset]
	overlap = qset.intersection(aset)
	return [float(len(overlap))] * 2
def cut(sentence):
	words = []
	for word in sentence.decode('utf-8').lower().split() :
	# for word in word_tokenize(sentence.decode('utf-8').lower()):
		if word not in stopwords.words('english') and word not in english_punctuations:
			try:
				word = stemmer.stem(word)
				# if word in va:
				words.append(word)
			except Exception  as e:
				print e
		# print words
		# print " ".join(words)
		return words
def idf_word_overlap(row):
	question = cut(row["question"])
	answer = cut(row["answer"])
	overlap = set(answer).intersection(set(question))
	idf_overlap = 0
	for word in overlap:
		if word in idf_words_dict:
			# small_idf_dict[word] = idf_words_dict[word]
			# if word.isdigit():
				# print word,idf_words_dict[word]
			idf_overlap += idf_words_dict[word]
	# print idf_overlap,len(overlap)
	idf_overlap = (idf_overlap + 1.0) / (len(overlap)+1.0)
	return idf_overlap
def overlap_index(question,answer,q_len,a_len,stopwords = []):
	qset = set([q for q in question if q not in stopwords])
	aset = set([a for a in answer if a not in stopwords])

	q_index = np.ones(q_len)
	a_index = np.ones(a_len)

	overlap = qset.intersection(aset)
	for i,q in enumerate(question):
		value = 0
		if q in overlap:
			value = 1
		q_index[i] = value
	for i,a in enumerate(answer):
		value = 0
		if a in overlap:
			value = 1
		a_index[i] = value
	return q_index,a_index
def idf_overlap(row,stopwords = []):
	question = row["question"].split()
	answer = row["answer"].split()
	qindex = [q for q in question if q not in stopwords]
	# print len(qindex)
	aindex = [a for a in answer if a not in stopwords]
	qset = set(qindex)
	# overlap = [a for a in aindex if a in qset]
	aset = set(aindex)
	# overlap = [q for q in qindex if q in aset]
	overlap = qset.intersection(aset)

	# get idf_overlap
	idf_overlap = 0
	for word in overlap:
		idf_overlap += idf_outer.get(word,0)
	return idf_overlap
def overlap_visualize():
    train,test,dev = load("trec",filter=False)
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
def get_feature():
	train,test,dev = load("trec",filter = False)
	test = test.reindex(np.random.permutation(test.index))

	test['pred'] = test.apply(overlap_jiabing,axis = 1)
	print evaluation.evaluationBypandas(test,test['pred'])


if __name__ == '__main__':
	# get_feature()
	data_processing()
	# random_result()
	# overlap_visualize()