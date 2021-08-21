### import and install necessary packages

import os
import re
import random
import glob
import sys
import pickle

import pandas as pd
import numpy as np

import time

# to install if you don't install yet
# !{sys.executable} -m pip install spacy
# !{sys.executable} -m spacy download en_core_web_sm

import spacy
nlp = spacy.load('en_core_web_sm')

# to install if you don't install yet
# !{sys.executable} -m pip install spacy-entity-linker  
nlp.add_pipe("entityLinker", last=True)   # to make use of the entityLinker





### purpose: process big file into separate text for each wikipage 
### input: each big file 
### output: the dataframe(columns= ['Directory','file_ID', 'file_title', 'file_text'])
def parse_segmt_file(content, dir_str):

    # the dataframe to store the separate file for each segmentation
    df_file = pd.DataFrame(columns= ['Directory','file_ID', 'file_title', 'file_text'])

    # predifined variables
    file_ID = ''
    file_title = ''
    file_text = ''
    # the tagger to use in for loop 
    start = -1
    end = -1

    for inx_i, tx in enumerate(content):
        # to find the begining of a seprate file
        res1 = re.match('<doc id="(.*)" .* title="(.*)">', tx)
        if res1 is not None:
            start = inx_i
            file_ID = res1[1]
            file_title = res1[2]

        # to find the end of a seprate file
        if re.match('</doc>', tx) is not None:
            end = inx_i
        if start!= -1 and end!= -1:
            file_text =' '.join(content[start+1:end])
            # reset the tagger
            start = -1
            end = -1

            # put the values into dataframe
            if len(file_text)>500:
                df_file = df_file.append({'Directory': dir_str, 'file_ID': file_ID, 'file_title': file_title, 'file_text': file_text}, ignore_index=True)


    return df_file






### purpose: find the index of the tagged NPs in a sentence
### input: (a tagged NP, a sentence tokenized in a list) 
### output: a tuple indicating the start and end indexes of this NPs in the tokenized list
def find_word_inx(ele, sents_ls):
    # initialize for test
    word_inx = (0,0)
    
    # check the length of element
    lenth = len(ele.split(' '))

    # the seed is not NP
    if lenth == 1:
        if ele in sents_ls:
            a = sents_ls.index(ele)
            word_inx = (a, a)

    # the seed is NP (2,3,.... and more tokens)
    elif lenth > 1:
        a_ls = []
        for i in range(lenth):
            eleHere = ele.split(' ')[i]
            if eleHere in sents_ls:
                a_ls.append(sents_ls.index(eleHere))
                word_inx = (a_ls[0], a_ls[-1])     

    # test for return
    if word_inx != (0,0):
        return word_inx
    else:
        return None


    
### purpose: used inside the function <twoEntites_sentence_file2>
### input: the list of searching NPs; the list of a sentence in tokens
### output: the index of NPs in token lists
def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))

    return results  


### purpose: recognized the NPs pairs in a sentence
### input: (a dataframe including each separate wikipage, a resulting dataframe , a list including the seed pairs) 
### output: a dataframe to store the sentences and other info. DataFrame(columns=['pairs', 'ele1_word_idx', 'ele2_word_idx', 'sentence', 'tokens', 'file_ID', 'file_title', 'SeedTF'])   
def twoEntites_sentence_file2(df_file, df_enwiki_causality, causality_pairs_list):
    
    for inx in range(len(df_file)):
        file = df_file.iloc[inx]['file_text']
        doc = nlp(file.lower())
        
        # returns all entities in the whole document
        all_linked_entities = doc._.linkedEntities

        # iterates over sentences and prints linked entities
        for sent in doc.sents:
            candid = []
            for entities in sent._.linkedEntities:
                #(1). to ensure that the text is the same as that in Wididata
                if entities.get_label() == entities.get_span().text:
                    candid.append(entities.get_label())
            
            tokens = [s.text for s in sent]
            ele1_2_word_idx = []
            SeedTF = False
            
            two_entites = []  
            # (2). if include the seed pairs, use the seed pairs
            for pairs in causality_pairs_list:
                if len(set(pairs).intersection(set(candid))) == 2:
                    two_entites = pairs
                    SeedTF = True
            
            #(3). only extract the elements that have the top two longest strings
            if not SeedTF:
                # remove the duplicate
                candid = list(set(candid))
                candid.sort(key = len)
                two_entites = candid[-2:]
                
            #(4). find the index of entities   
            if len(two_entites) == 2:

                for ele in two_entites:
                    res = find_sub_list(ele.split(' '), tokens)
                    if res != []:
                        ele1_2_word_idx.append(res[0])
                    else:
                        ele1_2_word_idx.append(None)
                df_enwiki_causality = df_enwiki_causality.append({'pairs': two_entites, 'ele1_word_idx': ele1_2_word_idx[0], 'ele2_word_idx': ele1_2_word_idx[1],
                                                                        'sentence': sent.text, 'tokens': [s.text for s in sent], 
                                                                        'file_ID': df_file.iloc[inx]['file_ID'], 'file_title': df_file.iloc[inx]['file_title'],
                                                                        'SeedTF': SeedTF},
                                                               ignore_index=True)
    return df_enwiki_causality




###!-------------------- main function --------------------!###


def main():

    print("---------------Procedure 02: pre-process the Wikipages Dataset--------------")
    
    path_here = os.getcwd()
    # wikipedia: Part of the whole Dataset (please see details in <download_data.sh>)
    enwiki_data = path_here + '/data/enwiki_20210601/text/'
    # get the seed pairs
    with open(path_here+'/res/causality_pairs_list.pickle', 'rb') as f:
        causality_pairs_list = pickle.load(f)

    print('-----------Attention: 100 segmentations, 60mins-150 mins required------------')
    print('-----------If time limited, please choose "FastMode" to skip this step------------')
    
    # the dataframe to store the sentences info with CORRECT causal pairs            
    df_enwiki_causality_AA = pd.DataFrame()

    # To proceed with each segmentation file
    dir_str = 'AA'
    for filename in glob.iglob(enwiki_data+dir_str+'/*',recursive = True):
        round_start = time.time()
        with open(filename, 'r') as f:
            content = f.readlines()
            content = [x.strip() for x in content] 
            print('-----------SegmentationFile: '+filename+'------------')

            # process segmentation into separate files with info
            df_file = parse_segmt_file(content, dir_str)
            print('separating this segmentation file into independent files ... ')

            # extract sentences where causal pairs appear
            df_enwiki_causality_AA = twoEntites_sentence_file2(df_file, df_enwiki_causality_AA, causality_pairs_list)
            print('------------Finished this file----------------')

        round_end = time.time()
        print('This round use '+ str((round_end-round_start) / 60) +'mins')


        # save to the disk 
        df_enwiki_causality_AA.to_csv(path_here + '/res/df_enwiki_causality_AA.csv')
        df_enwiki_causality_AA.to_pickle(path_here + '/res/df_enwiki_causality_AA.pkl')

    print("The processed WikiPages datasets are saved to -->  ./res/df_enwiki_causality_AA")
    
if __name__ == "__main__":
    main()



