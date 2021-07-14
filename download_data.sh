#!/bin/bash
set -euxo pipefail

# Check that we are running from the right directory.
if [ ! "${PWD##*/}" = "GSOC_RelationExtraction_github" ]; then
    echo "Script must be run from 'GSOC_RelationExtraction_github' directory" >&2
    exit 1
fi


#!---------------- download the datasets for Tast 01 -----------------!#
### the source link could also be found with manually download <https://drive.google.com/file/d/0B_jQiLugGTAkMDQ5ZjZiMTUtMzQ1Yy00YWNmLWJlZDYtOWY1ZDMwY2U4YjFk/view?sort=name&layout=list&num=50&resourcekey=0-k0OTSIGrF9UAcrTFfInlrw>

DATA_URL='https://docs.google.com/uc?export=download&id=0B_jQiLugGTAkMDQ5ZjZiMTUtMzQ1Yy00YWNmLWJlZDYtOWY1ZDMwY2U4YjFk'
FILES='SemEval2010_task8_all_data'
RELOAD=false

# Check if at least any file is missing. If so, reload all data.
if [ ! -e "data/SemEval2010_task8_all_data" ]; then
    RELOAD=true
fi

# if file is missing, download the new ones
if [ "$RELOAD" = true ]; then
    if [ -d "data/" ]; then rm -Rf "data/"; fi
    mkdir -p data
    wget --no-check-certificate $DATA_URL -O data.zip
    cd data/
    unzip ../data.zip
    rm ../data.zip
fi


#!----------------  download the datasets for Tast 02 ---------------- !#

# If this path cannot be found, please try this link to get the lastest version that are available <https://ftp.acc.umu.se/mirror/wikimedia.org/dumps/enwiki/> or you can find other resources in your closest mirror <https://meta.wikimedia.org/wiki/Mirroring_Wikimedia_project_XML_dumps#Current_Mirrors>

### the original steps to get the source file:
# 1). download this <https://ftp.acc.umu.se/mirror/wikimedia.org/dumps/enwiki/20210701/enwiki-20210701-pages-articles-multistream1.xml-p1p41242.bz2>
# 2). use the wikiextractor tool <https://pypi.org/project/wikiextractor/> to extract this bz2 file. (here i use the version wikiextractor-3.0.4)
# 3). get the folder containing the text

### the convenient approach
# Directly download the extracted text in this link <https://drive.google.com/file/d/190M19-Q7HL6VkKoRNJsDMGARo5nwo1Ir/view?usp=sharing>, i uploaded it to google drive after preprocessing in my side.


#! wait for DEBUGing (cannot download, Virus scan warning by Google drive) 
ENWIKI_URL='https://docs.google.com/uc?export=download&id=190M19-Q7HL6VkKoRNJsDMGARo5nwo1Ir'
FILES='enwiki_20210601'
RELOAD=false

# Check if at least any file is missing. If so, reload all data.
if [ ! -e "data/enwiki_20210601" ]; then
    RELOAD=true
fi

# if file is missing, download the new ones
if [ "$RELOAD" = true ]; then
    if [ -d "data/enwiki_20210601/" ]; then rm -Rf "data/enwiki_20210601/"; fi
    mkdir -p data/enwiki_20210601
    wget --no-check-certificate $ENWIKI_URL -O data.zip
    cd data/enwiki_20210601/
    unzip ../data.zip
    rm ../data.zip
fi






