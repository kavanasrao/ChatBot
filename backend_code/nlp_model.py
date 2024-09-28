import numpy as np 
import nltk

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer() 

nltk.download('punkt_tab')

nltk.download('punkt', download_dir='C:/Users/kavana s/nltk_data')


from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    tokens = nltk.word_tokenize(sentence)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]
    return tokens 
     

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [lemmatizer.lemmatize(W) for W in tokenized_sentence]
     
    bag = np.zeros(len(all_words),dtype=np.float64)
    for idx , W in enumerate(all_words):
        if W in tokenized_sentence:
            bag[idx] = 1.0
    return bag


import nltk
print(nltk.data.path)


