import pandas as pd
import ConfigParser
import os
import cPickle as pickle
import random
from nltk.corpus import stopwords
from nltk import stem
from nltk.tokenize import word_tokenize
import gensim
import numpy as np
import evaluation
from sklearn import linear_model
from sklearn import svm
import jieba
import pynlpir
import nltk
import re
from data_helpers import load
isEnglish = True
idf_words_dict = pickle.load(open('data/small_idf_dict','r'))
if isEnglish:
	print 'is English'
	stemmer = stem.lancaster.LancasterStemmer()
	english_punctuations = [',','.',':','?','(',')','[',']','!','@','#','%','&']
# using for overlap
def cut(sentence):
	if isEnglish:
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
		return words
# is chinese
	try:
		words= pynlpir.segment(sentence, pos_tagging=False)
	except:
		words= jieba.cut(str(sentence))
	words = [word for word in words if word not in stopwords and word in va]
	return words
def word_overlap(row):
	question = cut(row["question"]) 
	answer = cut(row["answer"])
	overlap = set(answer).intersection(set(question)) 
	return len(overlap)
# when and time
def special_Feature(row):
	# questions = row["question"].lower().split()
	isnumber = False
	answers = row["answer"].lower().split()
	pattern = re.compile(r'1\d{3}|20[012]\d')
	if 'when' in row['question'].lower():
		for answer in answers:
			match = pattern.match(answer)
			if match:
				isnumber = True
				break
	if isnumber:
		return 1		
	return 0

			
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
def get_features(df):
	df['overlap'] = df.apply(word_overlap,axis = 1)
	df['idf_overlap'] = df.apply(idf_word_overlap,axis = 1)
	df['special_feature'] = df.apply(special_Feature,axis = 1)
	names = list()
	names.append("overlap")
	names.append("idf_overlap")
	names.append('special_feature')
	df[names] = (df[names] -df[names].mean())/df[names].std(ddof=0)
	print names
	# exit()
	return names
def englishTest():
	train,test,dev = load("wiki",filter = True)
	q_max_sent_length = max(map(lambda x:len(x),train['question'].str.split()))
	a_max_sent_length = max(map(lambda x:len(x),train['answer'].str.split()))
	print 'train length',len(train)
	print 'test length', len(test)
	print 'dev length', len(dev)
	# test = test.reindex(np.random.permutation(test.index))
	train = train.reset_index()
	# test = test.reset_index()
	print 'load Data finished'
	columns1 = get_features(train)
	columns2 = get_features(test)
	common = [item for item in columns2 if item in columns1]
	print common
	# common = ['align', 'align_reverse', 'features_similarity', 'features_similarity_reverse']
	print 'save the small idf_dict'
	# pickle.dump(small_idf_dict,open('data/small_idf_dict','w'))
	x = train[common].fillna(0)
	y = train["flag"]
	test_x = test[common].fillna(0)
	# clf = linear_model.LinearRegression()
	clf = linear_model.LogisticRegression()
	# clf = svm.SVR()
	print x.head()
	# clf = GradientBoostingRegressor()
	# clf = tree.DecisionTreeRegressor()
	# clf = svm.SVR()
	clf.fit(x, y)
	print clf.coef_
	# predicted = clf.predict(test_x)
	predicted = clf.predict_proba(test_x)
	predicted = predicted[:,1]
	print len(predicted)
	print len(test)
	print (evaluation.evaluationBypandas(test,predicted))
if __name__ == '__main__':
	englishTest()