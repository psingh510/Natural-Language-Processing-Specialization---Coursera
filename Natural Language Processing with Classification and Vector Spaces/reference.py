#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import string
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer


# In[5]:


def preprocess_tweets(tweet):
    
    #removing characters
    tweets = re.sub(r'\$\w*','',tweet)
    tweets = re.sub(r'RT[\s]+','',tweets)
    tweets = re.sub(r'https?://[^\s\n\r]+','',tweets)
    tweets = re.sub(r'#','',tweets)
    
    #instantiate tokenizer class
    tokenizer = TweetTokenizer(preserve_case = False, strip_handles = True, reduce_len = True)
    tweet_tokens = tokenizer.tokenize(tweets)
    
    #importing english stop words
    stopwords_english = stopwords.words('english')
    clean_tweets = []
    
    #Instantiate stemming class
    stemmer = PorterStemmer()
    
    for word in tweet_tokens:
        if (word not in stopwords_english and word not in string.punctuation):
            stem_word = stemmer.stem(word)
            clean_tweets.append(stem_word)
        
    return clean_tweets 


# In[7]:


def build_frequency(tweets, label):
    
    label_list = np.squeeze(label).tolist()
    frequency = {}
    
    for y, tweet in zip(label_list,tweets):
        for word in preprocess_tweets(tweet):
            pair = (word, y)
            if pair in frequency:
                frequency[pair] += 1
            else:
                frequency[pair] = 1
                
    return frequency


def get_vectors(embeddings, words):
   
    return np.array([embeddings[word] for word in words])






