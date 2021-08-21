# Causal Relation Classifier

## Introduction
* Purposes
  * find the __cause-effect relations__ among entities in text by using deep learning techniques
  * e.g. Given a sentence, _"Suicide is one of the leading causes of death among pre-adolescents and teens."_, we expect to predict that entity __'suicide'__ has cause-effect relation to entity __'death'__.

* Problems
  * no widely used train set and test set
  * hard to choose the suitable entity pairs from a lot of entities in a sentence
  * complex to apply deep learning techniques for relation extraction task

* Solutions
  * <**01_seed_preparation.py**>: pre-process the dataset of [SemEval-2010 Task 8 Dataset](https://www.kaggle.com/drtoshi/semeval2010-task-8-dataset) into semi-structured format.
  * <**02_data_preparation.py**>: deal with partial WikiPages dump in English by using 'name entity recognition' spaCy tools, and store sentences into semi-structured format.
  * <**03_classification_models.py**>: train classifiers by applying different sentence embeddings to Logistic Regression Model and to LSTM, and evaluate the classification performance.
  * <**04_classification_bootstrapping.py**>: train the mentioned classifiers with bootstrapping techniques.


## One-Command Pipeline

The following command will install the packages according to the configuration file <__requirements.txt__>.
```bash
pip install -r requirements.txt
```

To run the complete pipeline, please use the command:

```bash
sh pipeline.sh [$1 Project_series] [$2 Dimensions_of_sentence_embeddings] [$3 Experiment_mode] [$4 Evaluation_times] [$5 Random_times]
```
| Parameter | Description | Type | Default |
| :--------:|-------------|------|--------:|
| $1 | The project's Series | String | Required
| $2 | Dimension of the sentence embeddings | Integer {50,100,200,300} | 50 |
| $3 | Experiment mode | String {'FastMode', 'CompleteMode'} | 'CompleteMode' |
| $4 | Evaluation times |Integer | 3 |
| $5 | Random times | Integer | 1 |

Examples:

```bash
sh pipeline.sh 'Test1'
```
```bash
sh pipeline.sh 'Test2' 50 'FastMode' 3 1
```

# Attention
* In this repository, the directory'./data' is empty. The datasets will be downloaded when you run the <__pipleline.sh__>
* For simplified, please just run 'FastMode'.
