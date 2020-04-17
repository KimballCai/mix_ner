#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''
@version:0.1
@author:Cai Qingpeng
@file: test.py
@time: 2020/3/18 7:30 PM
'''

import numpy as np
from flair.data import Corpus
from flair.datasets import ColumnCorpus
import pandas as pd

# define columns
columns = {0: 'text', 1: '_', 2: '_', 3: 'ner'}

# this is the folder in which train, test and dev files reside
data_folder = './data'  # /path/to/data/folder

# init a corpus using column format, data folder and the names of the train, dev and test files
corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file='train.txt',
                              test_file='test.txt',
                              dev_file='valid.txt')

print(corpus)

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary)


real = []

for sentence in corpus.test:
    for token in sentence.tokens:
        real.append(token.get_tag("ner").value)

from flair.models import SequenceTagger
labels = tag_dictionary.get_items()
print(labels)

# bert_model = SequenceTagger.load('./log/bert_20200319181345/best-model.pt')
# elmo_model = SequenceTagger.load('./log/elmo_20200319142948/best-model.pt')
# xlnet_model = SequenceTagger.load('./log/xlnet_20200319155116/best-model.pt')
# flair_model = SequenceTagger.load('./log/pool_flair_f_20200330163807/best-model.pt')

bert_64_model = SequenceTagger.load('./log/bert_64/best-model.pt')
bert_128_model = SequenceTagger.load('./log/bert_128/best-model.pt')
bert_512_model = SequenceTagger.load('./log/bert_512/best-model.pt')
# elmo_64_model = SequenceTagger.load('./log/elmo_64/best-model.pt')
# elmo_128_model = SequenceTagger.load('./log/elmo_128/best-model.pt')
# elmo_512_model = SequenceTagger.load('./log/elmo_512/best-model.pt')
# xlnet_64_model = SequenceTagger.load('./log/xlnet_64/best-model.pt')
# xlnet_128_model = SequenceTagger.load('./log/xlnet_128/best-model.pt')
# xlnet_512_model = SequenceTagger.load('./log/xlnet_512/best-model.pt')
# flair_64_model = SequenceTagger.load('./log/flair_64/best-model.pt')
# flair_128_model = SequenceTagger.load('./log/flair_128/best-model.pt')
# flair_512_model = SequenceTagger.load('./log/flair_512/best-model.pt')

def model_prediction(model):
    model_pred = []
    for sentence in corpus.test:
        model.predict(sentence)
        for token in sentence.tokens:
            model_pred.append(token.get_tag("ner").value)
    return model_pred


def cal_proba_score(model,sentence):
    model.predict(sentence,all_tag_prob=True)
    score = []
    for t_id, token in enumerate(sentence.tokens):
        # print(token.get_tag("ner").value)
        # print(token.get_tags_proba_dist("ner"))
        for index,item in enumerate(token.get_tags_proba_dist("ner")):
            # print(item.value)
            # print(item.score)
            score.append(item.score)
    return score

def get_whole_result(model):
    whole_results = []
    for i,sentence in enumerate(corpus.test):
        sen_result = cal_proba_score(model,sentence)
        whole_results.extend(sen_result)
    return whole_results

def get_mix_preds_by_result(results,method="avg"):
    if method == "avg":
        scores = np.mean(results,axis=0)
    elif method == "confidence":
        scores = np.max(results,axis=0)
    else:
        raise NotImplementedError(method)
        
    result = np.reshape(scores,(-1,len(labels)))
    
    id_result = np.argmax(result,axis=1)
    la_result = [tag_dictionary.get_item_for_index(i) for i in id_result]
    
    return la_result
        

bert_poss = get_whole_result(bert_64_model)
bert_result_list = pd.DataFrame(bert_poss)
bert_result_list.to_csv('result/prob/bert_64.csv')
print("Saving bert 64 result done")
bert_poss = get_whole_result(bert_128_model)
bert_result_list = pd.DataFrame(bert_poss)
bert_result_list.to_csv('result/prob/bert_128.csv')
print("Saving bert 128 result done")
bert_poss = get_whole_result(bert_512_model)
bert_result_list = pd.DataFrame(bert_poss)
bert_result_list.to_csv('result/prob/bert_512.csv')
print("Saving bert 512 result done")

# elmo_poss = get_whole_result(elmo_64_model)
# elmo_result_list = pd.DataFrame(elmo_poss)
# elmo_result_list.to_csv('result/prob/elmo_64.csv')
# print("Saving elmo 64 result done")
# elmo_poss = get_whole_result(elmo_128_model)
# elmo_result_list = pd.DataFrame(elmo_poss)
# elmo_result_list.to_csv('result/prob/elmo_128.csv')
# print("Saving elmo 128 result done")
# elmo_poss = get_whole_result(elmo_512_model)
# elmo_result_list = pd.DataFrame(elmo_poss)
# elmo_result_list.to_csv('result/prob/elmo_512.csv')
# print("Saving elmo 512 result done")

# xlnet_poss = get_whole_result(xlnet_64_model)
# xlnet_result_list = pd.DataFrame(xlnet_poss)
# xlnet_result_list.to_csv('result/prob/xlnet_64.csv')
# print("Saving xlnet 64 result done")
# xlnet_poss = get_whole_result(xlnet_128_model)
# xlnet_result_list = pd.DataFrame(xlnet_poss)
# xlnet_result_list.to_csv('result/prob/xlnet_128.csv')
# print("Saving xlnet 128 result done")
# xlnet_poss = get_whole_result(xlnet_512_model)
# xlnet_result_list = pd.DataFrame(xlnet_poss)
# xlnet_result_list.to_csv('result/prob/xlnet_512.csv')
# print("Saving xlnet 512 result done")

# flair_poss = get_whole_result(flair_64_model)
# flair_result_list = pd.DataFrame(flair_poss)
# flair_result_list.to_csv('result/prob/flair_64.csv')
# print("Saving flair 64 result done")
# flair_poss = get_whole_result(flair_128_model)
# flair_result_list = pd.DataFrame(flair_poss)
# flair_result_list.to_csv('result/prob/flair_128.csv')
# print("Saving flair 128 result done")
# flair_poss = get_whole_result(flair_512_model)
# flair_result_list = pd.DataFrame(flair_poss)
# flair_result_list.to_csv('result/prob/flair_512.csv')
# print("Saving flair 512 result done")
