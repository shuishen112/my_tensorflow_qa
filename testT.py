from multiprocessing import Pool
import math
import numpy as np
import datetime

def f(x):
	print x
	y = [1] * 10000000
	[math.exp(i) for i in y]
def g(x):
	print x
	y = np.ones(10000000)
	np.exp(y)
def formal(f,l):
	for i in l:
		f(i)

if __name__ == '__main__':
	p = Pool(4)
	begin = datetime.datetime.now()
	# formal(f,range(100))
	map(f,range(100))
	# p.map(f,range(100))
	end = datetime.datetime.now()
	print end - begin