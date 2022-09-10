# Neural extraction Framework @DBpedia
This is my GSoC’22 project on building Knowledge Graph using NLP.

|   Project Details     | |
|-------------|-------------|
| GSoC Project | [Neural Extraction Framework GSoC'22 @DBpedia](https://summerofcode.withgoogle.com/programs/2022/projects/HIqpMFb3)        |
| Mentors | [Tommaso Soru](https://github.com/mommi84), [Diego Moussallem](https://github.com/DiegoMoussallem), [Ziwei Xu](https://github.com/zoeNantes)|
| Blogs | [Neural Extraction Framework GSoC'22 by Ananya](https://ananyaiitbhilai.github.io/DBpedia_GSoC2022_Neural_Extraction_Framework) |



It uses the [REBEL model](https://huggingface.co/Babelscape/rebel-large) from the hugging face library. The REBEL model performs both NER and Relation Extraction. It takes input as a sentence and outputs a triple *{head, type, tail}*.

Summary of the Project tasks:
1. Scraped the text from the Wikipedia article
2. Perform Coreference Resolution
3. Tokenises articles into Sentences
4. REBEL model is run on each Sentence
5. The confidence level for each triple is calculated
6. Corresponding Head and Tail entities are matched to their respective URLs

## Installations

### For .ipynb
You can find the **neural_extraction_framework.ipynb** notebook. You can either use your local machine or Google Colab to run the notebook by cloning the repo.

Installation of the libraries used:
```
!pip install wikipedia
!pip install transformers
!pip install nltk
!pip install fuzzywuzzy
!pip install rdflib
!pip install SPARQLWrapper
```
Install spaCy and neuralcoref with correct versions
```
!pip install -U spacy==2.1.0 
!python -m spacy download en
!pip uninstall -y neuralcoref 
!pip install neuralcoref --no-binary neuralcoref
```
However, all the commands to install the dependencies is present in the notebook, you only have to run all the cells. In the `url_link` variable enter the URL of the Wikipedia article for which you want to extract the triples, and run all the cells. In the end, you will get a table consisting of the triple along with the confidence level of the relation between the entities. Here is an image of what the table would look like:


### For .py
Or you can use the **neural_extraction_framework.py** file. Here, the user have to input the URL of the Wikipedia article. To install the dependencies, you have to run shell file `cmd.sh` with the help of the following command:
```
bash cmd.sh
```
In this case, the user gives the input of the URL of the Wikipedia article for which it wants to extract the Triples.
```
python neural_extraction_framework.py
```

In case you are not able to install the dependencies in your local system, you can use the google colab and run the following commands in the cells:
```

```
```
!bash cmd.sh
```
```
!python neural_extraction_framework.py
```
*Please note: It only works for the English language Wikipedia articles and the Triple extraction is carried out only for the first 10 sentences since it takes some time to execute. But you can increase the number of sentences for which you want to extract the triples by changing the number of iterations to the length of the list containing all the sentences present in the article. You can also change the value of parameters:  "num_beams": 10,
    "num_return_sequences": 10*

## Future Goals
I would continue contributing to the Neural Extraction Framework Project
- Storing triples in a Graph Database
- If a pronoun text is anchored to a URL(relatively very few instances) then we miss out on those URLs while matching it in the coreference resolved text.  For this, we can use a list to store such instances.
- Making coreference resolution better
- Extracting all the correct URLs corresponding to the entities

I would love to build a model that could predict a new relation after accomplishing the above tasks.

Overall, it was an amazing and enriching experience for me, I got to learn a lot during GSoC. My all mentors were highly supportive, and knowledgeable and guided me along the path. I look forward to working and learning with them.