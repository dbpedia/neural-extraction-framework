import re
import json
import copy
import string
import numpy as np

def compare_clean_golden_ext_with_oie_ext(ext_g,ext_oie):
	ext_oie = ext_oie.split('\t')
	# print('here '*50)
	# print(ext_g, len(ext_g))
	# print(ext_oie, len(ext_oie))
	assert len(ext_g) == len(ext_oie)
	# sl = []
	bw = []
	for g,o in zip(ext_g,ext_oie):
		ol = o.split()
		gl = g.split()
		tbw = []
		i,j = 0,0
		# print(g)
		# print(o)
		while i < len(ol) and j < len(gl):
			# print(i,len(ol),j,len(gl))
			if ol[i]!=re.sub(r'\]|\[|\{[a-z]+\}','',gl[j]): # match failed
				# print('not matched')
				# print(ol[i],'vs',gl[j])
				if '[' == gl[j][0]:
					bracket_start, bracket_end = j, j
					while ']' not in gl[bracket_end]:
						bracket_end+=1
					if '{' in gl[bracket_end] and '}' in gl[bracket_end]:
						tbw.append(re.search(r'\{[a-z]+\}',gl[bracket_end])[0][1:-1])
					gl = gl[:bracket_start]+gl[bracket_end+1:]
					continue
				else:
					break
			else:
				# print('matched')
				# print(ol[i])
				i+=1
				j+=1
		if i == len(ol):
			while j != len(gl) and '[' == gl[j][0]:
				bracket_start, bracket_end = j, j
				while ']' not in gl[bracket_end]:
					bracket_end+=1
				if '{' in gl[bracket_end] and '}' in gl[bracket_end]:
					tbw.append(re.search(r'\{[a-z]+\}',gl[bracket_end])[0][1:-1])
				gl = gl[:bracket_start]+gl[bracket_end+1:]
			if j == len(gl):
				bw+=tbw
				# sl.append('satisfied')
			else:
				return 'not satisfied'
		else:
			return 'not satisfied'
	if bw:
		# print(bw)
		# input('wait')
		return 'satisfied but with '+','.join(bw)
	else:
		return 'satisfied'

def compare_raw_golden_ext_with_oie_ext(ext_golden, ext_oie, default_passive):
	ext_golden = ext_golden.split(' |OR| ')
	# print('raw',ext_golden)
	# print('raw',ext_oie)
	bl = []
	for ext_g in ext_golden:
		if ' <--{not allowed in passive}' in ext_g:
			passive = False
			ext_g = ext_g.replace(' <--{not allowed in passive}','')
		elif ' <--{allowed in passive}' in ext_g:
			passive = True
			ext_g = ext_g.replace(' <--{allowed in passive}','')
		elif ' <--{' in ext_g:
			print('Unknown command found in "'+ext_g+'"\nExiting')
			exit()
		else:
			passive = default_passive

		ext_g = ext_g.split(' --> ')
		b2 = ''
		b = compare_clean_golden_ext_with_oie_ext(ext_g, ext_oie)
		if b=='satisfied':
			return b
		if passive and b!='satisfied':
			t = ext_g[2]
			ext_g[2] = ext_g[0]
			ext_g[0] = t
			b2 = compare_clean_golden_ext_with_oie_ext(ext_g, ext_oie)
			if b2 == 'satisfied':
				return b2
		if 'satisfied but' in b:
			bl.append(b)
		else:
			bl.append(b2)
		
	bl2 = []
	for x in bl:
		if 'satisfied but' in x:
			bl2.append(x)
	if bl2:
		b = ' |OR| '.join(bl2) # now that I think hard, I realize that there will never be a situation where one extraction satisfies more than one pattern connected by |OR|
	else:
		b = 'not satisfied'
	return b

def n_extractions_in_smallest_cluster(golden_dict, n_sent):
	ext_g = golden_dict[n_sent]
	n = 0
	for cluster_no in ext_g.keys():
		n = max(len(ext_g[cluster_no]['essential']),n)
	return n

def calc_metrics(gold, exts, default_passive = True, show = False):
	golden_dict = {}
	essential_exts, compensating_dict, cluster_number, cluster_dict, sentence_number = [], {}, '', {}, ''

	for i,line in enumerate(gold):
		if 'sent_id:' in line:
			if sentence_number:
				ext_dict = {'essential':essential_exts, 'compensatory': compensating_dict}
				cluster_dict['cluster '+cluster_number] = ext_dict
				golden_dict['sent '+sentence_number] = cluster_dict
				essential_exts, compensating_dict, cluster_number, cluster_dict = [], {}, '', {}
				# print(json.dumps(golden_dict,indent=2,ensure_ascii=False).encode('utf8').decode() )
				# input('wait')
			sentence_number = re.search(r'sent_id:\d+',line)[0][8:]
			golden_dict['s'+sentence_number+' txt'] = line.split('\t')[1]
		elif '------ Cluster' in line:
			if cluster_number:
				ext_dict = {'essential':essential_exts, 'compensatory': compensating_dict}
				cluster_dict['cluster '+cluster_number] = ext_dict
				essential_exts, compensating_dict = [], {}
			cluster_number = re.search(r'\d+',line)[0]
		elif re.search(r'\{[a-z]\}',line[:4]):
			compensating_dict[line[1]] = line[4:]
		elif '='*20 not in line:
			essential_exts.append(line)

	ext_dict = {'essential':essential_exts, 'compensatory': compensating_dict}
	cluster_dict['cluster '+cluster_number] = ext_dict
	essential_exts, compensating_dict = [], {}
	golden_dict['sent '+sentence_number] = cluster_dict
	essential_exts, compensating_dict, cluster_number, cluster_dict = [], {}, '', {}

	if show:
		pass
		# print('Golden dictionary is')
		# print(json.dumps(golden_dict,indent=2,ensure_ascii=False).encode('utf8').decode()[:500]+'\n\t... ... ...'*3)

	# --- Gathering extractions ---
	ext_oie = {}
	for e in exts:
		e = e.split('\t')
		sno = e[0]
		e = re.sub(' +',' ','\t'.join(e[1:]))
		e = e.translate(str.maketrans('', '', string.punctuation+'।'))
		e = re.sub(' +',' ',e)
		try:
			if e not in ext_oie[sno]: # removes duplicates
				ext_oie[sno].append(e)
		except Exception as ex:
			ext_oie[sno] = [e]

	# print(ext_oie)
	if show:
		print('\nExtractions to evaluate are')
		print(json.dumps(ext_oie,indent=2,ensure_ascii=False).encode('utf8').decode()[:500]+'\n\t... ... ...'*3 )
		print('Total sents',len(ext_oie.keys()))

	golden_state_dict = copy.deepcopy(golden_dict)

	tpl, fpl, fnl = [], [], []

	# --- Populating the state dict ---
	for k in ext_oie.keys():
		ext_l = ext_oie[k] # ext_l is extractions of one sentence
		sno = k
		sent_dict = golden_dict['sent '+sno]
		state_dict = golden_state_dict['sent '+sno]
		tp, fp = 0, len(ext_l)
		for e in ext_l:
			tp_dict = {e:False}
			for cluster_no in sent_dict.keys():
				# print('-'*100)
				# print('sent',k,'cluster',cluster_no,'oie',e)
				for i,ext_g in enumerate(sent_dict[cluster_no]['essential']):
					if 'satisfied' not in state_dict[cluster_no]['essential'][i]:
						state_dict[cluster_no]['essential'][i] = 'not satisfied' # filling "not satisfied" by default
					elif state_dict[cluster_no]['essential'][i] == 'satisfied':
						continue # already satisfied, hence do not check this # it adds one additional feature i.e. if one extraction satisfies a particular pattern, then another cannot do it. Hence repetitive extractions are penalized.
					curr_state = compare_raw_golden_ext_with_oie_ext(ext_g, e, default_passive)
					if curr_state != 'not satisfied': # i.e. the extraction matched
						tp_dict[e] = True
					if curr_state == 'satisfied':
						state_dict[cluster_no]['essential'][i] = curr_state
					elif 'satisfied but' in curr_state:
						if 'satisfied but' in state_dict[cluster_no]['essential'][i]:
							state_dict[cluster_no]['essential'][i] += '|AND|'+curr_state
						else:
							state_dict[cluster_no]['essential'][i] = curr_state
					# else:
					# 	state_dict[cluster_no]['essential'][i] = curr_state

				for ck in sent_dict[cluster_no]['compensatory'].keys():
					ext_g = sent_dict[cluster_no]['compensatory'][ck]
					if 'satisfied' not in state_dict[cluster_no]['compensatory'][ck]:
						state_dict[cluster_no]['compensatory'][ck] = 'not satisfied'
					elif state_dict[cluster_no]['compensatory'][ck] == 'satisfied':
						continue
					curr_state = compare_raw_golden_ext_with_oie_ext(ext_g, e, default_passive)
					if curr_state != 'not satisfied': # i.e. the extraction matched
						tp_dict[e] = True
					if curr_state == 'satisfied':
						state_dict[cluster_no]['compensatory'][ck] = curr_state
					elif 'satisfied but' in curr_state:
						if 'satisfied but' in state_dict[cluster_no]['compensatory'][ck]:
							state_dict[cluster_no]['compensatory'][ck] += '|AND|'+curr_state
						else:
							state_dict[cluster_no]['compensatory'][ck] = curr_state
			if tp_dict[e]:
				tp+=1
			if show:
				print('The state of golden dict after processing of this extraction ("'+e+'"): ',sno)
				print(json.dumps(state_dict,indent=2,ensure_ascii=False).encode('utf8').decode())
		tpl.append(tp)
		fpl.append(fp-tp)
		if show:
			# print(fpl)
			# print(tp,fp)
			print('After processing all the extractions')
			print(ext_l)
			print('The state of golden dict is (of this sentence)')
			print(json.dumps(state_dict,indent=2,ensure_ascii=False).encode('utf8').decode())

	if show:
		pass
		# print('\nState of golden dictionary after processing extractions is')
		# print(json.dumps(golden_state_dict,indent=2,ensure_ascii=False).encode('utf8').decode()[:500]+'\n\t... ... ...'*3 )


	def fn_sb(cd,cel,fn=0):
		i = 0
		while i < len(cel):
			ce = cel[i]
			if 'not satisfied' == cd[ce]:
				fn+=1
				cd[ce] = 'X'
			elif 'satisfied but' in cd[ce]:
				cel2 = cd[ce].split()[-1].split(',')
				fn = fn_sb(cd,cel2,fn)
			i+=1
		return fn


	for sno in ext_oie.keys():
		# --- Take one cluster at a time, and select the minimum number of FN ---
		temp_fn = []
		for cluster_no in golden_state_dict['sent '+sno].keys():
			# print(cluster_no)
			# input('wait')
			cluster = golden_state_dict['sent '+sno][cluster_no]
			fn = 0
			for e in cluster['essential']:
				if e == 'not satisfied':
					fn+=1
				if 'satisfied but' in e:
					tfnl = []
					for e2 in e.split('|AND|'):
						cel = e2.split()[-1].split(',')
						tfnl.append(fn_sb(cluster['compensatory'].copy(),cel))
					fn += min(tfnl)

					# cel = e.split()[-1].split(',')
					# fn += fn_sb(cluster['compensatory'].copy(),cel)
			temp_fn.append(fn)
		fnl.append(min(temp_fn))
		# if 'sent ' in sno:
			# fn = 0
			# cluster_with_most_satisfied = (0,'cluster 1')
			# for cluster_no in golden_state_dict[sno].keys():
			# 	cluster = golden_state_dict[sno][cluster_no]
			# 	satno = 0
			# 	for e in cluster['essential']:
			# 		if 'satisfied' == e or 'satisfied but' in e:
			# 			satno+=1
			# 	for k in cluster['compensatory'].keys():
			# 		if 'satisfied but' in cluster['compensatory'][k] or 'satisfied' == cluster['compensatory'][k]:
			# 			satno+=1
			# 	if satno > cluster_with_most_satisfied[0]:
			# 		cluster_with_most_satisfied = (satno, cluster_no)
			# print('cluster_with_most_satisfied',cluster_with_most_satisfied[1],cluster_with_most_satisfied[0])

			# cluster = golden_state_dict[sno][cluster_with_most_satisfied[1]]
			# for e in cluster['essential']:
			# 	if e == 'not satisfied':
			# 		fn+=1
			# 	if 'satisfied but' in e:
			# 		cel = e.split()[-1].split(',')
			# 		fn += fn_sb(cluster['compensatory'].copy(),cel)
			# fnl.append(fn)

	missing_fn = []
	for sent_n in golden_dict.keys():
		if sent_n.split()[-1] not in ext_oie.keys() and 'txt' not in sent_n:
			missing_fn.append(n_extractions_in_smallest_cluster(golden_dict,sent_n))



	# p = []
	# r = []
	# fl = []
	# for tp, fp, fn in zip(tpl,fpl,fnl):
	# 	if tp == 0:
	# 		p.append(0)
	# 		r.append(0)
	# 		fl.append(0)
	# 	else:
	# 		precision = tp/(tp+fp)
	# 		recall = tp/(tp+fn)
	# 		p.append(precision)
	# 		r.append(recall)
	# 		fl.append(2*(precision*recall)/(precision+recall))

	# print('Recall',sum(r)/len(r))
	# print('Precision',sum(p)/len(p))
	# print('F-score',sum(fl)/len(fl))

	p = sum(tpl)/(sum(tpl)+sum(fpl)) if (sum(tpl)+sum(fpl)) != 0 else 0
	r = sum(tpl)/(sum(tpl)+sum(fnl)+sum(missing_fn)) if (sum(tpl)+sum(fnl)+sum(missing_fn)) != 0 else 0
	f = 2*p*r/(p+r) if (p+r) != 0 else 0

	print('  ',[b for a in ([x]*10 for x in range(8)) for b in a][1:])
	print('  ',(list(np.arange(1,10))+[0])*8)
	print('TP',tpl,len(tpl))
	print('FP',fpl)
	print('FN',fnl)
	print('Missing FNs',missing_fn)

	# Calculate and print final totals
	total_tp = sum(tpl)
	total_fp = sum(fpl)
	total_fn = sum(fnl) + sum(missing_fn)
	
	print(f'\n=== FINAL METRICS ===')
	print(f'Total TP (True Positives): {total_tp}')
	print(f'Total FP (False Positives): {total_fp}')
	print(f'Total FN (False Negatives): {total_fn}')
	print(f'  - Regular FN: {sum(fnl)}')
	print(f'  - Missing FN: {sum(missing_fn)}')
	print(f'Total Extractions: {total_tp + total_fp}')
	print(f'Total Expected: {total_tp + total_fn}')

	print('Recall',r)
	print('Precision',p)
	print('F-score',f)

	return r,p,f, golden_state_dict, tpl, fpl, fnl, missing_fn, ext_oie, golden_dict

def write_detailed_analysis(analysis_file_path, golden_dict, ext_oie, golden_state_dict, tpl, fpl, fnl, missing_fn):
    sno_list = list(ext_oie.keys())
    with open(analysis_file_path, 'w', encoding='utf-8') as af:
        af.write("Detailed Analysis Report\n")
        af.write("="*50 + "\n\n")

        # Create a map from sno to its index for tpl, fpl, fnl
        sno_to_idx = {sno: i for i, sno in enumerate(sno_list)}

        # All sentences from the gold standard
        gold_sents = {k.split(' ')[1] for k in golden_dict.keys() if k.startswith('sent ')}

        for sno in sorted(gold_sents, key=int):
            af.write(f"Sentence ID: {sno}\n")
            af.write(f"Sentence Text: {golden_dict.get('s' + sno + ' txt', 'N/A')}\n")
            
            if sno in sno_to_idx:
                idx = sno_to_idx[sno]
                af.write(f"Metrics: TP={tpl[idx]}, FP={fpl[idx]}, FN={fnl[idx]}\n")
            
                # Model Extractions Analysis
                af.write("\n--- Model Extractions ---\n")
                if sno in ext_oie:
                    # This part needs the per-extraction TP/FP status, which is not directly available.
                    # For now, just listing them.
                    for model_ext in ext_oie[sno]:
                        af.write(f"- {model_ext}\n")
                else:
                    af.write("No extractions from model for this sentence.\n")

                # Gold Standard Analysis
                af.write("\n--- Gold Standard Analysis ---\n")
                sent_state_dict = golden_state_dict.get('sent '+sno)
                if sent_state_dict:
                    for cluster_no in sorted(sent_state_dict.keys()):
                        cluster_data = sent_state_dict[cluster_no]
                        af.write(f"\nCluster: {cluster_no}\n")
                        af.write("  Essential Extractions:\n")
                        for i, essential in enumerate(cluster_data['essential']):
                            gold_pattern = golden_dict['sent ' + sno][cluster_no]['essential'][i]
                            status = essential if isinstance(essential, str) else "satisfied" # Simplified
                            af.write(f"    - Gold: {gold_pattern}\n")
                            af.write(f"      Status: {status}\n")

                        if cluster_data.get('compensatory'):
                            af.write("  Compensatory Extractions:\n")
                            for key, compensatory in cluster_data['compensatory'].items():
                                gold_pattern = golden_dict['sent ' + sno][cluster_no]['compensatory'][key]
                                status = compensatory if isinstance(compensatory, str) else "satisfied" # Simplified
                                af.write(f"    - Gold ({key}): {gold_pattern}\n")
                                af.write(f"      Status: {status}\n")
                else:
                    af.write("No state information available for this sentence.\n")

            else: # Sentences missed by model
                af.write("This sentence was not present in the model's output.\n")
                af.write("\n--- Gold Standard ---\n")
                sent_gold_dict = golden_dict.get('sent ' + sno, {})
                for cluster_no, cluster_data in sent_gold_dict.items():
                    af.write(f"\nCluster: {cluster_no}\n")
                    af.write("  Essential Extractions:\n")
                    for essential in cluster_data.get('essential', []):
                        af.write(f"    - {essential}\n")
                    if cluster_data.get('compensatory'):
                        af.write("  Compensatory Extractions:\n")
                        for key, comp in cluster_data.get('compensatory', {}).items():
                            af.write(f"    - ({key}) {comp}\n")

            af.write("\n" + "="*50 + "\n\n")

## ----- a small stub to calculate hindi_benchie scores on the english sample ----- 
# file = open('extractions/english_explicit.text','r')
# exts = [x.strip() for x in file.readlines()]
# file.close()
# file = open('english_benchie_gold.txt','r')
# gold = [x.strip() for x in file.readlines()]
# file.close()
# print(calc_metrics(gold, exts))
# exit()
import os
import glob
ml = glob.glob('extractions/formatted_triplets_6000.txt')
default_passive = True
show = False

nms, pl, rl, fl = [], [], [], []

file = open('hindi_benchie_gold.txt','r')
gold = [x.strip() for x in file.readlines()]
file.close()

analysis_dir = 'detailed_analysis'
if not os.path.exists(analysis_dir):
    os.makedirs(analysis_dir)

for fname in ml:
	# if 'indie' not in fname and 'keshav' not in fname:
	# 	continue
	print('-'*100,fname,'-'*100)
	nms.append(fname.replace('extractions/',''))
	file = open(fname,'r')
	exts = [x.strip() for x in file.readlines()]
	file.close()

	r, p, f, golden_state_dict, tpl, fpl, fnl, missing_fn, ext_oie, golden_dict = calc_metrics(gold, exts, show=show)
	rl.append(r)
	pl.append(p)
	fl.append(f)

	# After calc_metrics, generate the detailed analysis file
	analysis_filename = os.path.join(analysis_dir, os.path.basename(fname).replace('.txt', '_analysis.txt'))
	
	write_detailed_analysis(analysis_filename, golden_dict, ext_oie, golden_state_dict, tpl, fpl, fnl, missing_fn)
	print(f"Detailed analysis report generated at: {analysis_filename}")


import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()

x = np.arange(1,2+(len(nms)-1)*5,5)
y = rl
plt.bar(x,y)

x = np.arange(2,3+(len(nms)-1)*5,5)
y = pl
plt.bar(x,y)

x = np.arange(3,4+(len(nms)-1)*5,5)
y = fl
plt.bar(x,y)

x = np.arange(2,3+(len(nms)-1)*5,5)
plt.xticks(x,nms)

plt.legend(['recall','precision','f-score'])

# plt.show()

