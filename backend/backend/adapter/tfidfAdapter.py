from chatterbot.logic import LogicAdapter
import pandas
import gensim
import collections
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import string
from collections import defaultdict
from gensim import corpora, models, similarities
from gensim.models.doc2vec import Doc2Vec, LabeledSentence  
from collections import Counter

#you can set the tags for different flows and then return the corresponding response when the users question matches the tags in tn tag list.

class TfIdfAdapter(LogicAdapter):
    def __init__(self, **kwargs):
        super(TfIdfAdapter, self).__init__(**kwargs)
    
    def split(self, document):
        #split the sentence into word and remove the stop word
        stoplist=set('for a of the and to in at after with'.split())  
        texts=[word for word in document.lower().split() if word not in stoplist]
        return texts

    def splitWord(self, documents):
        #split the sentence into word and remove the stop word
        stoplist=set('for a of the and to in at after with'.split())  
        texts=[[word for word in document.lower().split() if word not in stoplist] for document in documents]
        return texts

    def euclidean_distance(self, x,y):
        return math.sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))
 
    def manhattan_distance(self, x,y):
        return sum(abs(a-b) for a,b in zip(x,y))
     
    def jaccard_similarity(self, x,y):
        intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
        union_cardinality = len(set.union(*[set(x), set(y)]))
        return intersection_cardinality/float(union_cardinality)

    def cosine(self, vector1, vector2):
        tmp = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if tmp == 0:
            return 0
        else:
            return float(np.dot(vector1,vector2) / tmp)
        
    def KL(self, a, b):
        a = np.asarray(a, dtype=np.float)
        b = np.asarray(b, dtype=np.float)
        return np.sum(np.where(a != 0, a * np.log(a / b), 0))

    def tfidf_model(self, statement):
        #create dummy data 
        statement = statement.text
        tags = ['finTech financial finance', 'regTech regulations regulation', 'debt coinselling broking credit references lending peer CCL consumer']

        tags = self.splitWord(tags)
        statement = self.split(statement)

         # get the total number of words in the corpus/ assume each word appear once in the body for the calculation of idf
        words_corpus = []
        for tag in tags:
            tag = list(set(tag))
            words_corpus += tag

        # get the idf list of every word in the corpus, idf is the number of times each word occurs in the document
        tags_len = len(tags)
        idf_list = Counter(words_corpus)
        for word in idf_list:
            idf_list[word] = math.log(tags_len/idf_list[word])

        #get the tfidf value for the tags
        tags_tfidf = {}
        tags_norm = {}
        for i in range(0,len(tags)):
            tfidf_list = {};
            tag_word_count = Counter(tags[i])
            word_number = len(tags[i])
            for word in tag_word_count:
                tfidf_list[word] = tag_word_count[word]/word_number * idf_list[word]

            tags_tfidf[i] = tfidf_list
            tags_norm[i] = np.linalg.norm(list(tfidf_list.values()))

        #get the tfidf value for the statement
        statement_tfidf = {};
        statement_word_count = Counter(statement)
        word_number = len(statement)
        for word in statement_word_count:
            statement_tfidf[word] = statement_word_count[word]/word_number * idf_list[word]

        #calculate similarity between the tags and statement
        cos_sim_list = []
        euclidean_dist_list = []
        kl_list = []
        word_count_list = []


        for i in range(0,len(tags)):
            cos_sim = 0
            euclidean_dist = 0
            kl = 0
            word_count = 0
            tag_tfidf = tags_tfidf[i]
            tag_norm =  tags_norm[i]

            for word in statement_tfidf:
                if word in tag_tfidf:
                    vec1 = statement_tfidf[word]
                    vec2 = tag_tfidf[word]
                    cos_sim += vec1*vec2
                    euclidean_dist += pow((vec1 - vec2),2)
                    kl += vec1 * np.log(vec1 / vec2)
                    word_count += 1
                else:
                    euclidean_dist += pow(statement_tfidf[word], 2)

            for word in tag_tfidf:
                if word not in statement_tfidf:
                    euclidean_dist += pow(tag_tfidf[word], 2)

            euclidean_dist = math.sqrt(euclidean_dist)
            cos_sim /= (np.linalg.norm(list(statement_tfidf.values())) * tag_norm)
            cos_sim_list.append(cos_sim)
            euclidean_dist_list.append(euclidean_dist)
            kl_list.append(kl)
            word_count_list.append(word_count)       

        return np.argmax(cos_sim_list);

    def can_process(self, statement):
        """
        Return true if the input statement contains the tags
        """
        statement.text = statement.text.translate(str.maketrans('','',string.punctuation))
        #add the tags for different flows here
        tags =  [['finTech', 'financial', 'finance'], ['regTech','regulations','regulation'], ['debt', 'coinselling', 'broking', 'credit', 'references', 'lending', 'peer', 'CCL', 'consumer']]
        for tag in tags:
            for x in tag:
                if x in statement.text.split():
                    print(x)
                    print('bingo')
                    return True

        return False

    
    def process(self, statement):
        from chatterbot.conversation import Statement
         #remove punctuation
        statement.text = statement.text.translate(str.maketrans('','',string.punctuation))
        confidence = 1
        #add the response for different flow.
        response_list = ['you can go to the finTech page', 'you can go to the regTech page', 'you can go to the consumer credit page']
        index = self.tfidf_model(statement)
       

        selected_statement = Statement(response_list[index])
        selected_statement.confidence = 1

        return selected_statement







