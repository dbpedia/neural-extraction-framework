import pickle

file = open('hindi_benchie_gold.txt','r')
content = file.readlines()
file.close()

ml = []
file = open('sents.txt','w')
for line in content:
	if 'sent_id:' in line:
			file.write(line)
			ml.append(' '.join(line.split('\t')[1:]).strip())
file.close()

# print(ml[:3])
# file = open('sents.pkl','wb')
# pickle.dump(ml,file)
# file.close()

# file = open('sents.txt','w')
# file.write('\n'.join(ml))
# file.close()
