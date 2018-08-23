import os
import sys
import random
import insuranceqa_data as insuranceqa
import pandas as pd
import numpy as np
from gensim import corpora, models, similarities
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from stop_words import get_stop_words


def splitWordByLibrary(documents):
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


 
def preprocess():    
    train_doc_question = pd.read_csv('../data/FiQA_train_question_doc_final.tsv', sep='\t')
    train_question = pd.read_csv('../data/FiQA_train_question_final.tsv', sep='\t')
    train_doc = pd.read_csv('../data/FiQA_train_doc_final.tsv', sep='\t')

    qdic = train_question.set_index('qid').T.to_dict('list')
    docdic = train_doc.set_index('docid').T.to_dict('list')

    question_id_list = train_doc_question['qid']
    doc_id_list = train_doc_question['docid']

    questions = []
    docs = []

    for i in range(0, len(question_id_list)):
        # question = train_question[train_question.qid == question_id_list[i]]['question'].values[0]
        # doc = train_doc[train_doc.docid == doc_id_list[i]]['doc'].values[0]
        question = qdic[question_id_list[i]][1]
        doc = docdic[doc_id_list[i]][1]
        questions.append(question)
        docs.append(doc)

    texts = np.hstack((questions,docs))
    texts = splitWordByLibrary(texts)
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

preprocess()