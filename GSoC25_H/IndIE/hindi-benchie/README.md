# hindi-benchie

📊 Presenting an automated evaluation benchmark 🧮 to calculate precision, recall, and F1 scores for automatically extracted triples from 112 Hindi sentences using diverse Open Information Extraction (OIE) tools. 🌟 This benchmark accompanies the research paper titled "IndIE: A Multilingual Open Information Extraction Tool For Indic Languages," which has been accepted for publication in the Findings of IJCNLP-AACL 2023. 📚🔍

## Requirements

Python 3

## Running

* Sentences are given in sents.txt file.
* Run the desired triple extractor tool on each sentence and save the output in the following format:
     * sent-num \<tab> head \<tab> relation \<tab> tail
* Save the output in a .txt file and save it in the ```extractions``` folder.
* Run: ```python code.py```