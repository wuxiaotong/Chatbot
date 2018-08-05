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

SOME_FIXED_SEED = 43
np.random.seed(SOME_FIXED_SEED)


def euclidean_distance(x,y):
    return math.sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))
 
def manhattan_distance(x,y):
    return sum(abs(a-b) for a,b in zip(x,y))
 
def jaccard_similarity(x,y):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)

def cosine(vector1, vector2):
    tmp = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    if tmp == 0:
        return 0
    else:
        return float(np.dot(vector1,vector2) / tmp)
    
def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

def load_data():
	with open("../data/consumer_credit_data.json",'r') as load_f:
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

def splitWord(documents):
    #split the sentence into word and remove the stop word
    texts = []
    stoplist=set('for a of the and to in at after with do i was am an Do its so need on if be were are is who we fca'.split())  
    for document in documents:
        document = document.translate(str.maketrans('','',string.punctuation))
        tmp = []
        for word in document.lower().split():
            if word not in stoplist:
                tmp.append(word)
        texts.append(tmp)
    return texts

def lda_model():
    statement = "we've been told we don’t need a licence to sell credit but been the rates are our business would pay the finance companies would be a lot higher than to be not entirely regulated  but if we get a licence and get regulated that’s gonna bring the rates of the finance down"
    # statement = "They've already told me, that unless I'm doing a small proportion of my overall advice in debt counselling then I'm not covered"
    statement = statement.translate(str.maketrans('','',string.punctuation))
    data = load_data()
    id_list = data[1]
    tag_list= data[0]
    texts = splitWordByLibrary(tag_list)
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    ldamodel = LdaModel(corpus, num_topics=len(corpus), id2word = dictionary) 
    cos_sim_list = []
    corpus_len = len(corpus)
    for i in range(0, corpus_len):
        new_vec = dictionary.doc2bow(statement.lower().split())
        dict1 = dict(ldamodel[new_vec])
        dict2 = dict(ldamodel[corpus[i]])
        vec2 = np.zeros(corpus_len)
        vec1 = np.zeros(corpus_len)
        for a in dict2:
            vec2[a] = dict2[a]
        for a in dict1:
            vec1[a] = dict1[a]
        cos_sim_list.append(cosine(vec1, vec2))

    largest_index = np.argmax(cos_sim_list)
    print(largest_index)
    print(cos_sim_list[largest_index])   
    print(tag_list[largest_index])

def tfidf_model():
    #create dummy data 
    # statement = statement.text
    statement = "I've been told because of the kind of work I may do from time to time that I have to differentiate what would fall under CCL and what wouldn't"
    # statement = "They've already told me, that unless I'm doing a small proportion of my overall advice in debt counselling then I'm not covered"
    statement = "I sell cars through finance, I've been told I must get FCA limited permissions to be able to do that"
    statement = "I've been told that I need to apply for a licence to sell my goods If we sell consumers credit, can we offer 12 months interest free credit without the licence?"
    statement = "we've been told we don’t need a licence to sell credit but been the rates are our business would pay the finance companies would be a lot higher than to be not entirely regulated  but if we get a licence and get regulated that’s gonna bring the rates of the finance down"
    # statement = "They've already told me, that unless I'm doing a small proportion of my overall advice in debt counselling then I'm not covered"
    statement = statement.translate(str.maketrans('','',string.punctuation))


    data = load_data()
    id_list = data[1]
    tag_list= data[0]
    
    tags = splitWord(tag_list)

    cos_sim_list = []
    euclidean_distance_list = []
    manhattan_distance_list = []
    jaccard_similarity_list = []
    word_count_list = []

    dictionary = corpora.Dictionary(tags)
    corpus = [dictionary.doc2bow(tag) for tag in tags]
    corpus_len = len(dictionary)
    tfidf = models.TfidfModel(corpus)
    index = similarities.MatrixSimilarity(tfidf[corpus], num_features=len(dictionary))

    for i in range(0, len(tags)):
        new_vec = dictionary.doc2bow(statement.lower().split())
        dict1 = dict(tfidf[new_vec])
        dict2 = dict(tfidf[corpus[i]])

        vec2 = np.zeros(corpus_len)
        vec1 = np.zeros(corpus_len)

        for a in dict2:
            vec2[a] = dict2[a]
        for a in dict1:
            vec1[a] = dict1[a]
        cos_sim_list.append(cosine(vec1, vec2))
        euclidean_distance_list.append(euclidean_distance(vec1, vec2))
        jaccard_similarity_list.append(jaccard_similarity(vec1, vec2))
        manhattan_distance_list.append(manhattan_distance(vec1, vec2))
        # word_count_list.append(len(dict1)+len(dict2))
    largest_index = np.argmax(cos_sim_list)
    print(largest_index)
    print(cos_sim_list[largest_index])
    # print('input')
    # print(statement)
    # print('tag')
    print(tag_list[largest_index])
    # if(cos_sim_list[largest_index] < 0.18):
    #     print('all flow')
  
def doc_sim_model():
    statement = "I've been told because of the kind of work I may do from time to time that I have to differentiate what would fall under CCL and what wouldn't"
    statement = "I sell cars through finance, I've been told I must get FCA limited permissions to be able to do that"
    statement = "I've been told that I need to apply for a licence to sell my goods If we sell consumers credit, can we offer 12 months interest free credit without the licence?"
    statement = "we've been told we don’t need a licence to sell credit but been the rates are our business would pay the finance companies would be a lot higher than to be not entirely regulated  but if we get a licence and get regulated that’s gonna bring the rates of the finance down"
    # statement = "They've already told me, that unless I'm doing a small proportion of my overall advice in debt counselling then I'm not covered"
    statement = statement.translate(str.maketrans('','',string.punctuation))
    data = load_data()
    id_list = data[1]
    tag_list = data[0]
    tags = splitWord(tag_list)
    dictionary = corpora.Dictionary(tags)
    corpus = [dictionary.doc2bow(tag) for tag in tags]
    similarity = Similarity('-Similarity-index', corpus, num_features=400)
    test_statement = dictionary.doc2bow(statement.lower().split())
    similarity.num_best = 1
    most_similar = similarity[test_statement][0]
    print(most_similar)
    if len(most_similar) > 0:
        index = most_similar[0]    
        print(tag_list[index])

def doc_model():
    statement = "I've been told because of the kind of work I may do from time to time that I have to differentiate what would fall under CCL and what wouldn't"
    statement = statement.translate(str.maketrans('','',string.punctuation))
    data = load_data()
    id_list = data[1]
    tag_list = data[0]
    texts = splitWord(tag_list)
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]
    model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)
    infer_vector = model.infer_vector(statement.lower().split())
    print(model.docvecs.most_similar([infer_vector], topn = 1))  



tfidf_model()
doc_sim_model()
lda_model()

