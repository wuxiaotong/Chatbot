from chatterbot.logic import LogicAdapter
import numpy as np
import math
import pandas as pd
import string
import json
from gensim import corpora, models, similarities
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.similarities import Similarity
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from gensim.models import LdaModel
from stop_words import get_stop_words
from pandas import DataFrame
from difflib import SequenceMatcher
from chatterbot.conversation import Statement

SOME_FIXED_SEED = 43
np.random.seed(SOME_FIXED_SEED)

#you can use the document similarity adatper to match the users' question to the corresponding region in the consumer credit flow

class DocumentSimialrityAdapter(LogicAdapter):
    def __init__(self, **kwargs):
        super(DocumentSimialrityAdapter, self).__init__(**kwargs)
        
    def can_process(self, statement):
        """
        Return true if the input statement contains the tags
        """ 
        return True

    def load_data(self):
        with open("./backend/data/FCA/consumer_credit_data.json",'r') as load_f:
            load_dict = json.load(load_f, encoding='utf-8')
            id_list = []
            data_list = []

            for item in load_dict['questions']:
                id_list.append(item['_id'])
                data_list.append(item['label'] + ' ' +item['externalComment']+ ' ' + item.get('tip', '')+ ' ' + item.get('internalComment', ''))

            for item in load_dict['licences']:
                if not(item.get('externalComment', '').find('This activity is exempted. You do not need to be authorised.')):
                    id_list.append(item['_id'])
                    data_list.append(item['label'] + ' ' + item.get('externalComment', ''))

        return [data_list, id_list]

    def splitWordByLibrary(self, documents):
        texts = []
        tokenizer = RegexpTokenizer(r'\w+')
        en_stop = get_stop_words('en')
        p_stemmer = PorterStemmer()

        for i in documents:
            # clean and tokenize document string
            raw = i.lower()
            tokens = tokenizer.tokenize(raw)

            # remove stop words from tokens
            stopped_tokens = [i for i in tokens if not i in en_stop]

            # stem tokens
            stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

            # add tokens to list
            texts.append(stemmed_tokens)
        return texts

    def lsi_model(self, statement, data):
        id_list = data[1]
        tag_list = data[0]

        tags = self.splitWordByLibrary(tag_list)
        dictionary = corpora.Dictionary(tags)
        corpus = [dictionary.doc2bow(tag) for tag in tags]
        tfidf = models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]

        lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=len(tag_list))
        index = similarities.MatrixSimilarity(lsi[corpus_tfidf]) 
        statement = self.splitWordByLibrary([statement])[0]
     
        test_statement = dictionary.doc2bow(statement)
        vec_lsi = lsi[test_statement]
        sims = index[tfidf[vec_lsi]]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        largest_index = sims[0][0]
        return [tag_list[largest_index], id_list[largest_index], sims[0][1]]

    def difflib_model(self, statement, data):
       
        id_list = data[1]
        tag_list= data[0]
        tags = self.splitWordByLibrary(tag_list)
        statement = self.splitWordByLibrary([statement])[0]

        largest_similarity = 0
        largest_index = 0
        for i in range(0, len(tags)):
            confidence = self.compare_statements(Statement(statement), Statement(tags[i]))

            if confidence > largest_similarity:
                largest_similarity = confidence
                largest_index = i
            
        return [tag_list[largest_index], id_list[largest_index], confidence]
    
    def process(self, statement):
         #remove punctuation
        data = self.load_data()
        
        lsi_result, lsi_id, lsi_confidence = self.lsi_model(statement.text, data)
        difflib_result, difflib_id, difflib_confidence = self.difflib_model(statement.text, data)
        
        selected_statement = Statement(lsi_result)
        selected_statement.confidence = lsi_confidence

        return selected_statement
  

