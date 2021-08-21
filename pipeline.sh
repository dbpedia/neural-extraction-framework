#!/bin/bash
set -euxo pipefail


# $1 -- The project's Series -- String -- Required
# $2 -- Dimension of the sentence embeddings -- Integer [50|100|200|300] -- Optional, 50 by default
# $3 -- Experiment mode -- String ['FastMode', 'CompleteMode'] -- Optional, 'CompleteMode' by default
# $4 -- Evaluation times -- Integer -- Optional, 3 by default
# $5 -- Random times -- Integer -- Optional, 1 by default


if [ ! -n "$1" ] ;then
    echo "you have not input a project's series in string, e.g. 'project1'!"
else
    project_series=$1
    echo "The project's series will be set to $1"
fi


if [ ! -n "$2" ] ;then
    vector_size=50
elif [[ ! $2 =~ ^[0-9]*$ ]]; then
    echo "Please enter an integer [50|100|200|300] to the second parameter to set the dimension of Glove Embeddings;"
    vector_size=50
elif [ $2 == 50 ]; then
    vector_size=50
    echo "The dimension of GloVe embeddings is set to $vector_size"
elif [ $2 == 100 ]; then
    vector_size=100
    echo "The dimension of GloVe embeddings is set to $vector_size"
elif [ $2 == 200 ]; then
    vector_size=200
    echo "The dimension of GloVe embeddings is set to $vector_size"
elif [ $2 == 300 ]; then
    vector_size=300
    echo "The dimension of GloVe embeddings is set to $vector_size"
else
    vector_size=50
    echo "The dimension of GloVe embeddings is set to $vector_size"
fi


if [ ! -n "$3" ] ;then
    experim_mode='CompleteMode'
elif [ $3 != "FastMode" ] && [ $3 != "CompleteMode" ]; then
    echo "Please choose the experiment mode with 'FastMode' or 'CompleteMode'. "
elif [ $3 == "FastMode" ]; then
    experim_mode='FastMode'
    echo "The experiment mode is set to $experim_mode"
elif [ $3 == "CompleteMode" ]; then
    experim_mode='CompleteMode'
    echo "The experiment mode is set to $experim_mode"
else
    experim_mode='CompleteMode'
    echo "The experiment mode is set to $experim_mode"
fi


if [ ! -n "$4" ] ;then
    evaluation_times=3
elif [[ ! $4 =~ ^[0-9]*$ ]]; then
    echo "Please enter an integer [ <=10 recommended ] to the fourth parameter to set the number of evaluation times"
elif [ "$4" -le 10 ]; then
    evaluation_times=$4
    echo "The times of evaluation is set to $evaluation_times"
else
    evaluation_times=3
fi


if [ ! -n "$5" ] ;then
    random_times=1
elif [[ ! $5 =~ ^[0-9]*$ ]]; then
    echo "Please enter an integer [ <=10 recommended ] to the fifth parameter to set the random times"
elif [ "$5" -le 10 ]; then
    random_times=$5
    echo "The random times is set to $random_times"
else
    random_times=1
fi



# Check that we are running from the right directory.
if [ ! "${PWD##*/}" = "GSOC_RelationExtraction_github" ]; then
    echo "Script must be run from 'GSOC_RelationExtraction_github' directory" >&2
    exit 1
fi


#!---------------- download the datasets for Procedure 01 -----------------!#
### the source link could also be found with manually download <https://drive.google.com/file/d/0B_jQiLugGTAkMDQ5ZjZiMTUtMzQ1Yy00YWNmLWJlZDYtOWY1ZDMwY2U4YjFk/view?sort=name&layout=list&num=50&resourcekey=0-k0OTSIGrF9UAcrTFfInlrw>

DATA_GD_id='1MJFcuL6_3-ctvv93cQtKfZQrGZS5GAlT'
RELOAD=false

# Check if at least any file is missing. If so, reload all data.
if [ ! -e "data/SemEval2010_task8_all_data" ]; then
    RELOAD=true
fi

# if file is missing, download the new ones
if [ "$RELOAD" = true ]; then
    if [ -d "data/" ]; then rm -Rf "data/"; fi
    mkdir -p data
    gdown --id $DATA_GD_id -O data.zip
    cd data/
    unzip ../data.zip
    rm ../data.zip
    cd .. 
fi


#!----------------  download the datasets for Procedure 02  ---------------- !#

# If this path cannot be found, please try this link to get the lastest version that are available <https://ftp.acc.umu.se/mirror/wikimedia.org/dumps/enwiki/> or you can find other resources in your closest mirror <https://meta.wikimedia.org/wiki/Mirroring_Wikimedia_project_XML_dumps#Current_Mirrors>

### the original steps to get the source file:
# 1). download this <https://ftp.acc.umu.se/mirror/wikimedia.org/dumps/enwiki/20210701/enwiki-20210701-pages-articles-multistream1.xml-p1p41242.bz2>
# 2). use the wikiextractor tool <https://pypi.org/project/wikiextractor/> to extract this bz2 file. (here i use the version wikiextractor-3.0.4)
# 3). get the folder containing the text

### the convenient approach
# Directly download the extracted text in this link <https://drive.google.com/file/d/190M19-Q7HL6VkKoRNJsDMGARo5nwo1Ir/view?usp=sharing>, i uploaded it to google drive after preprocessing in my side.


DATA_GD_id='190M19-Q7HL6VkKoRNJsDMGARo5nwo1Ir'
RELOAD=false

# Check if at least any file is missing. If so, reload all data.
if [ ! -e "data/enwiki_20210601" ]; then
    RELOAD=true
fi

# if file is missing, download the new ones
if [ "$RELOAD" = true ]; then
    if [ -d "data/enwiki_20210601/" ]; then rm -Rf "data/enwiki_20210601/"; fi
    mkdir -p data/enwiki_20210601
    cd data/enwiki_20210601/
    gdown --id $DATA_GD_id -O data.zip
    unzip ./data.zip
    rm ./data.zip
    cd ..
    cd ..
fi


#!----------------  run Procedures  ---------------- !#

#1. proecedure 01 to get the seed information from SemEval dataset
python 01_seed_preparation.py

# 2. proecedure 02 to get the semi-structured information from WikiPage dataset
prepro_exist="./res/df_enwiki_causality_AA.pkl"

if [ $experim_mode == "FastMode" ] ; then

    #!----------------  download the post-procesed datasets of Procedure 02 if required ---------------- !#
    ### If this method cannot be executed, directly download the extracted text in this link <https://drive.google.com/file/d/1Oaqg1mnnGTrk_OKnbULzd1c6BPDdDy3f/view?usp=sharing>, i uploaded it to google drive after preprocessing in my side.
    DATA_GD_id='1Oaqg1mnnGTrk_OKnbULzd1c6BPDdDy3f'
    RELOAD=false

    # Check if at least this file is existing. 
    if [ ! -e $prepro_exist ]; then
        RELOAD=true
    else
        echo "--------Procedure 02: The post-processed dataset exists--------"
    fi
    # the file is missing, download the new ones
    if [ "$RELOAD" = true ]; then
        cd ./res
        gdown --id $DATA_GD_id -O data.zip
        unzip ./data.zip
        rm ./data.zip
        cd ..
    fi

elif [ $experim_mode == "CompleteMode" ] ; then
    python 02_data_preparation.py
fi


#3. proecedure 03 to train classifiers on those datasets
python 03_classification_models.py $project_series $vector_size $evaluation_times


#4. proecedure 04 to train classifiers with bootstrapping
python 04_classification_bootstrapping.py $project_series $vector_size $evaluation_times $random_times


echo "----------------  END  ----------------"
