# IndIE

This is the code for the paper titled "IndIE: A Multilingual Open Information Extraction Tool For Indic Languages" accepted in the Findings of IJCNLP-AACL 2023.

You can check out the live deployment of the IndIE [here](http://103.25.231.59:80).

## Installation

* Git clone.
* Make a new virtual environment.
    * Upgrade pip by ```pip install -U pip```
* Install the necessary libraries by using the following command:
```pip install -r requirements.txt```
    * If you face any difficulty in installing any library then we recommend to install it without version number. Hence, pip will install from the latest version.
    * However, we do not recommend this for stanza library because a different version of stanza library will yield in different dependency parse trees [[source]](https://github.com/stanfordnlp/stanza/issues/990).

## Download Models

Download the relevant files from [here](https://drive.google.com/file/d/1UqOUdeK96m6EabI-cg2EeBz6p3IwrPZ6/view?usp=sharing).

Then place the files in the directories such that your directory structure looks like this:

```
.
├── chunking
│   ├── chunking_model.py
│   ├── crf_chunker.py
│   ├── data
│   │   └── my_tagset_BI.bin
│   └── state_dicts
│       └── model
│           ├── 26_epoch_4.pth.tar
│           └── sklearn_crf_model_v2_pos_mapped_2.pkl
```

## Memory needed

The amount of memory needed varies depending upon the number of languages on which you wish to run your triple extraction.

1) All languages (Hindi/Tamil/Telugu/Urdu)
    * GPU present
        * ~6GB on CPU/RAM
        * ~6GB on GPU
    * GPU absent
        * ~7GB on CPU/RAM
2) Only one language
    * GPU present
        * ~3GB on CPU/RAM
        * ~4GB on GPU
    * GPU absent
        * ~3GB on CPU/RAM


## Extracting triples

Specify the language and list of strings in the ```main.py``` file. 

On GPU, make sure you have same device order on nvidia-smi and PCI bus. Command: ```export CUDA_DEVICE_ORDER=PCI_BUS_ID```

Run

```CUDA_VISIBLE_DEVICES=0 python main.py```

where ```0``` is your GPU ID. The code also runs in absence of GPU but takes a little longer. In order to run the code only on CPU, simply omit the GPU ID.

## Citation

```Will be updated```


## Workflow for GSoC25_H
1. Update hyperparameters in main.py for running the kind of extractions:

```json
'use_llm': False,  # True for running entire MDT output through LLM
	'llm_fallback': True,  # set to True for running only on sentences which did not get any output from original rule based extractions
	'llm_enhancement': True,  # set to True for running rule based extraction + llm extraction 
	'llm_filter_mode': False,  # set to True for running llm based filter on the output of all the above strategies
	'llm_model': 'gemma3:12b-it-qat'  
```


2. Generate extractions using ```python main.py```

3. Convert the *.h5 generated file to tab sepearted extractions using ```python convert.py```

4. Copy the file the IndIE/hindi-benchie/extractions folder and update the code.py file with correct file path and run ```python code.py```.






