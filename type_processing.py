#coding=utf8
import pynlpir
import cPickle as pickle
pynlpir.open()
ner_dict = dict()
data_dir = 'data/nlpcc/'
count = 0
for filename in ["train_raw","test_raw",'submit_raw']:
	with open(data_dir + filename + '.txt') as f:
		for e,line in enumerate(f):
			splits = line.split('\t')
			q = splits[0]
			a = splits[1]
			try:
				segments = pynlpir.segment(a,pos_names = 'child')
			except Exception as ex:
				count += 1
				print ex
			word_type = []
			for seg in segments:
				if seg[1] == 'organization/group name':
					word_type.append(seg[1])
				elif seg[1] == 'personal name' or seg[1] == 'transcribed personal name':
					word_type.append(seg[1])
				elif seg[1] == 'toponym' or seg[1] == 'locative word' or seg[1] == 'transcribed toponym':
					word_type.append(seg[1])
			if len(word_type) > 0:
				ner_dict[a] = set(word_type)
			if e % 1000 == 0:
				print e
print 'encoding error',count
pickle.dump(ner_dict,open('ner_dict','w'))