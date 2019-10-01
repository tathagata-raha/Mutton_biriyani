
# Megathon_qualcomm

## Problem Statement

Identification of papers/websites/documents with context related to a given context or abstract

## How to run the file

- store the test data in the following csv files
- - abstract.csv : list of all the abstracts in csv format
- - text.csv     : list of all the full text of the research papers in csv format
- Run the following commands, while in this directory

```
pip install -r requirements.txt
python3 -m spacy download en
python3 main.py abstract.csv text.csv 
```

## Inputs and Constraints

1) Abstract csv with N abstracts
2) Full text csv with M full text articles

## Outputs

Returns/creates another similarity_matrix.csv which will have N rows and M columns where each cell (i,j) represents a similarity score between the abstract i and article j.  Each similarity score should be between 0 and 1.

## APPROACH TO SOLVE THE PROBLEM

- Prepare the text for topic modelling
- Generating topics using LDA models
- Correlating the topics with their abstracts
- Generating the similarity matrix
- Generating the correlation matrix

### Prepare the text for topic modelling

- Clean the text data obtained from the abstract and the full research paper, by using tokenizer() made by ourselves, to return a list of tokens 
- Using WordNetLemmatizer, get the root word of each of the token in the list
- Remove all the stopwords
- Then read line by line from the abstract.txt and text.txt files and tokenize the sentences.

### Generating topics using LDA models

- Using a tool, gensim, for topic modelling, assign topics to a particular piece of text document
- From this data, create a dictionary corpus based on the abstract.txt, convert to a bag of words corpus and save the dictionary and corpus for future use.
- Using this corpus and dictionary, gensim generates an LDA model that generates 5 topics related with the document. Each topic is characterized by 4 or more keywords and their weights.
- Next, we can use this LDA model the find the significance of each topic for the main document (that contains the full text of the research paper)

### Correlating the topics with their abstracts

- Using the LDA model trained previously, measure the significance of each topic generated for the main document, with the abstracts.

### Generating the similarity matrix

- We then generate the similarity matrix of each abstract with each document by multiplying the significance of each of the topics with the abstract and the significance of the same topic in the text document and then take the sum. 

(Note: We expect that if some topic has maximum significance with the document and the same topic has maximum significance with a particular abstract, then the product of the similarity values of that topic would be high, and hence the abstract would most probably belong to the document.)

### Generating the correlation matrix

- The correlation matrix is given by the similarity matrix.
- We have normalised so that max=1
- The output correlation matrix is loaded on an qualcomm.csv file.

## TESTING DATA
- We tested this model on a set of 70 document texts and 70 corresponding abstracts, which we obtained by web scraping from various sources.

