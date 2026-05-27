
'''
this file generates benchie compatible extractions from munnwar data
sirf munnwar wale data ke liye applicable hai
'''

import pandas as pd
df = pd.read_hdf('../../allData/indie-extractions/extractions_run6_hindi_benchie.h5',key='hi')
# file = open('indie_explicit.txt','w')
# for i, row in df.iterrows():
# 	ml = []
# 	for e in row['extractions']:
# 		ml += e
# 	for m in ml:
# 		file.write(str(i+1)+'\t'+m[0]+'\t'+m[1]+'\t'+m[2]+'\n')
# file.close()

df2 = pd.read_hdf('../../allData/indie-extractions/munnwar.h5',key='hi')
print(df2)
df3 = pd.read_hdf('../../allData/indie-extractions/munnwar_rand.h5',key='hi')
df4 = pd.concat([df2,df3]).reset_index(drop=True)
df4['sentence'] = df4['sentence'].apply(lambda d: d.strip())
print(df4)
df4.at[1,'sentence'] = "अखिल भारतीय पुलिस डयूटी मीट ( 1958 से ) में अंगुलि चिह्न विज्ञान प्रतियोगिता आयोजित करना ."
df4.at[2,'sentence'] = "केन्द्रीय सरकार के विभागों एवं भारत सरकार के उपक्रमों द्वारा भेजे गए विवादित अंगुलि चिह्नों का परीक्षण करना ."


file = open('sents.txt','r')
content = file.readlines()
file.close()

file = open('extractions/benchie_faruqui.txt','w') 
for i, sent in enumerate(content):
	sent = sent.split('\t')[1].strip()
	for m in df4.loc[list(df4['sentence'] == sent).index(True)]['extractions']:
		file.write(str(i+1)+'\t'+m[0]+'\t'+m[1]+'\t'+m[2]+'\n')
	# print('<'+df4.loc[i]['sentence']+'>')
	# input('wait')
file.close()