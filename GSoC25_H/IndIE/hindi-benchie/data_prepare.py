#download from https://www.kaggle.com/datasets/shankkumar/multilingualopenrelations15 

import pandas as pd
from tqdm import tqdm
import random, os, numpy as np
import copy
# import pickle

hyper_params = {'rseed':123}

os.environ['PYTHONHASHSEED'] = str(hyper_params['rseed'])
np.random.seed(hyper_params['rseed'])
random.seed(hyper_params['rseed'])

for lang in ['hi','ur','ta','te']:
	file = open({'hi':'hindi','ur':'urdu','ta':'tamil','te':'telugu'}[lang],'r')
	content = file.readlines()
	file.close()

	df = pd.DataFrame([],columns=['source','sentence','extractions'])

	d = dict()
	d2 = dict()
	sset = set()

	for i,line in tqdm(enumerate(content),total=len(content)):
		line = line.strip().split(' ||| ')
		if len(line) != 9:
			pass
			print(i+1)
			input('wait')
		else:
			try:
				d[line[1]] += [[line[2],line[3],line[4]]]
			except Exception as e:
				d[line[1]] = [[line[2],line[3],line[4]]]
				d2[line[1]] = line[0]
				sset.add(line[1])

	random_sents = copy.deepcopy(list(d.keys()))
	random.shuffle(random_sents)
	# print(random_sents[:100])
	# input('wait2')
	random_sents = random_sents[:500]
	i = 0
	for k in tqdm(random_sents, desc='Generating dataframe'):
		df.at[i,'source'] = d2[k]
		df.at[i,'sentence'] = k
		df.at[i,'extractions'] = d[k]
		i+=1

	print(df)
	print(df.loc[0]['sentence'])
	print(df.loc[0]['extractions'])

	df.to_hdf('munnwar_rand.h5',key=lang,mode='a')
	df.to_csv(lang+'_500.csv',index=None)
