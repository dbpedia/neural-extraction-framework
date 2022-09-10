### import necessary packages
import os
import pandas as pd
import numpy as np
import re
import random
import pickle




### purpose: get the causality sentences/pairs and other sentences/pairs from the tagged text and write them into dataframe
### input: the path of tagged text
### output: the dataframe to store causal pairs DataFrame(columns=['SentID', 'Cause','Effect','Label', 'Sent'])

def getSentSemEval(path_semEval2010): 

    # read the dataset
    with open(path_semEval2010, 'r') as f:
        content = f.readlines()
        content = [x.strip() for x in content] 


    #------------- get the causal pairs and negative pairs in dataset -----------#
    pattern_causal = 'Cause-Effect\((e.),(e.)\)'
    pattern_other_ref = '^\S*-\S*\((e.),(e.)\)$'
    pattern_e1 = '.*<e1>(.*)</e1>.*'        
    pattern_e2 = '.*<e2>(.*)</e2>.*'
    pattern_sentID = '(\d+)\\t.*'

    sent_list = []
    sent_id_list = []
    pair_list = []
    label_list = []

    for inx_l, lines in enumerate(content):

        # sentence content   
        res1 = re.match('\d*\\t\"(.*)\"', lines)
        if res1 is not None: 
            #get rid of other noise symbols of this sentence
            sent = re.match('\d*\\t\"(.*)\"', lines)[1]
            #delete the tags <e > in sentences
            sent = re.sub("</?e[1-2]>", "", sent)  
            sent_list.append(sent)

            #sentence id
            sent_id = re.match("(\d.*)\\t", lines)[1]
            sent_id_list.append(int(sent_id))

            # extract entities
            res_e1 = re.match(pattern_e1, lines)[1]
            res_e2 = re.match(pattern_e2, lines)[1]

            ### the next line
            # causal pairs
            res2 = re.match(pattern_causal, content[inx_l+1]) 
            if res2 is not None:
                # cause part + effect part (e1 or e2)
                if res2[1] == 'e1':
                    res_cause = res_e1
                    res_effect = res_e2
                if res2[1] == 'e2':
                    res_cause = res_e2
                    res_effect = res_e1
                # append the e1 then e2
                pair_list.append([res_cause.lower(), res_effect.lower()])
                label_list.append(1)
            else:
                pair_list.append([res_e1.lower(), res_e2.lower()])
                label_list.append(0)


    ### put all into df
    df_pairs = pd.DataFrame(columns=['SentID', 'Cause','Effect','Label', 'Sent'])
    df_pairs['SentID'] = sent_id_list
    df_pairs['Cause'] = [i[0] for i in pair_list]
    df_pairs['Effect'] = [i[1] for i in pair_list]
    df_pairs['Label'] = label_list
    df_pairs['Sent'] = sent_list

    # firstly extract postive rows
    df_pairs_p = df_pairs[df_pairs['Label'] == 1]
    # secondly extract negative rows
    df_pairs_n = df_pairs[df_pairs['Label'] == 0].sample(n = len(df_pairs_p))
    # converge the two df together
    df_pairsPN = pd.concat([df_pairs_p, df_pairs_n])
    
    return df_pairsPN
    

    
def main():

    print("---------------Procedure 01: prepare seed causal pairs from SemEval Dataset--------------")
    
    ### define the path of dataset
    path_here = os.getcwd()
    path_semEval2010 = path_here +'/data/SemEval2010_task8_all_data/'
    # training pairs
    path_semEval2010_train = path_semEval2010 +'SemEval2010_task8_training/TRAIN_FILE.TXT'
    # test pairs
    path_semEval2010_test = path_semEval2010 +'SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT'


    # get the casual sentences/pairs from train files and test files of semEval2010
    df_train_semEval = getSentSemEval(path_semEval2010_train)
    df_test_semEval = getSentSemEval(path_semEval2010_test)

    # save to 
    df_train_semEval.to_csv(path_here + '/res/df_train_semEval.csv')
    df_train_semEval.to_pickle(path_here + '/res/df_train_semEval.pkl')
    df_test_semEval.to_csv(path_here + '/res/df_test_semEval.csv')
    df_test_semEval.to_pickle(path_here + '/res/df_test_semEval.pkl')
    
    # get the seed pairs in train set
    causality_pairs_list = []
    for inx_r, row in df_train_semEval[df_train_semEval['Label']==1].iterrows():
        causality_pairs_list.append([row['Cause'], row['Effect']])
    with open(path_here+'/res/causality_pairs_list.pickle', 'wb') as f:
        pickle.dump(causality_pairs_list, f)
    

    print("The processed SemEval datasets are saved to -->  ./res/df_train_semEval and ./res/df_test_semEval")

    print("--------------------------Procedure 01: Finished-------------------------")
if __name__ == "__main__":
    main()

    