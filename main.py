import pandas as pd
import numpy
import sys
import spacy
import pickle
from gensim import corpora
import random
from spacy.lang.en import English
import nltk

from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer 

def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    

def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)


def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens


# for text_values_array in [abs_val]:
# #     topic_generator(text_values_array,abs_val)
#     import random
#     text_data = []   
# #    print (str(text_values_array[0][0]))


def topic_generator(txt,j_big):
    text_data = [] 
    txt_list = txt.split(".")
    var_size=1-40/len(txt_list) #suggests an appropriate size for the no of lines to be selected
#     print (txt_list[0:5])
# #     print(txt)
    for i in txt_list:
        tokens = prepare_text_for_lda(i)
        #     print(tokens)
        if random.random() > var_size:
        #print(tokens)
            text_data.append(tokens)
        #     print(text_data)
    
    dictionary = corpora.Dictionary(text_data) #generates a Dictionary of similar words
    corpus = [dictionary.doc2bow(text) for text in text_data]
    
    pickle.dump(corpus, open('corpus.pkl', 'wb'))
    dictionary.save('dictionary.gensim')
    import gensim
    NUM_TOPICS = 5 #we give 5 topics to choose from each text file
    
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
    ldamodel.save('model5.gensim')
    topics = ldamodel.print_topics(num_words=4)
    new_doc = txt    
    new_doc = prepare_text_for_lda(new_doc) #cleaning the text
    new_doc_bow = dictionary.doc2bow(new_doc)
#    print(new_doc_bow)
    list_mod=ldamodel.get_document_topics(new_doc_bow)
    list_mod=bal_zero(list_mod,NUM_TOPICS)
#    print(list_mod)
    list_mod_txt=list_mod
    list_mod_abs=[]

    final_sum_array = []
    for j in [j_big]:
        sum_list = []
        for k in range(abstract.shape[0]):   #iterating over each abstract data
#             print (j[k][0])
            new_doc = j[k][0]    
            new_doc = prepare_text_for_lda(new_doc)
            new_doc_bow = dictionary.doc2bow(new_doc)
            #print(new_doc_bow)
            list_mod=ldamodel.get_document_topics(new_doc_bow)
            list_mod=bal_zero(list_mod,NUM_TOPICS) #to balance out empty topics with priority 0
            # print(list_mod)
            list_mod_abs.append(list_mod)
            sum = 0
            for i in range(len(list_mod)):
                #print(">>>>>>>>>>>>><<<<<<<<<<<<<<<")
                #print(list_mod[i][1] * list_mod_txt[i][1])
                sum += list_mod[i][1] * list_mod_txt[i][1]
            #print("iuiuiuiuiuiu")
            #print(sum)
            sum_list.append(sum)
      
        max_list=max(sum_list)
        min_list=min(sum_list)
        
        for i in range(len(sum_list)):
            sum_list[i] = (sum_list[i]) / (max_list)
            #sum_list[i] = (sum_list[i] - min_list) / (max_list - min_list)
            
         #print(">>>>>>>>>>>>SUM LIST<<<<<<<<<<<<")
        return(sum_list) 

        final_sum_array.append(sum_list)
        
        # print("max of list : ")
        # print(max(sum_list))
        # print("min")
        # print(min(sum_list))
        
        # print("")
            
        # print("softmax list><><><><><><><><><><><><><>")
        # print(softmax(sum_list))


        

        
    # print(list_mod_abs)
def bal_zero(list_mod,no_of_topics):
    temp_list=[]
    for i in list_mod:
        temp_list.append(i[0])
    for i in range(no_of_topics):
        if i not in temp_list:
            list_mod.append((i,0))
    return list_mod 


# import numpy as np

# def softmax(x):
#     """Compute softmax values for each sets of scores in x."""
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum()

#-----------------------------main---------------------------------------

#checking for command line inputs
if(len(sys.argv) != 3):
    print('Invalid usage.')
    print('Correct usage: python3 main.py abstract.csv text.csv') 
abstract = pd.read_csv(sys.argv[1], header=None)
text = pd.read_csv(sys.argv[2], header=None)

abstract.dropna(inplace=True) #getting rid of null values

text.dropna(inplace=True)


spacy.load('en')

parser = English() #using spacy's parser for English

# nltk.download('wordnet')

# nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english')) #this is for filtering out English stopwords

abs_val = abstract.values
text_val = text.values
#Here we generate the final 2-D matrix iterating over each text value 

L = []
for text_values_array in [text_val]:
    for i in range(text.shape[0]):
        L.append(topic_generator(str(text_values_array[i][0]),abs_val))
# print(L)
print("Topic generated")
numpy_array = numpy.asarray(L).transpose()
#save to csv
numpy.savetxt("similarity_matrix.csv", numpy_array, delimiter=",")








