#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''
@version:0.1
@author:Cai Qingpeng
@file: test.py
@time: 2020/3/18 7:30 PM
'''



import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from flair.data import Corpus
from flair.datasets import ColumnCorpus

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

# 4. initialize embeddings
from flair.embeddings import ELMoEmbeddings
from flair.embeddings import BertEmbeddings
elmo_embedding = ELMoEmbeddings("small")

# 5. initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=elmo_embedding,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)

model = SequenceTagger.load('./log/elmo/best-model.pt')

from conlleval import evaluate

pred = []
real = []

for sentence in corpus.test:
    for token in sentence.tokens:
        real.append(token.get_tag("ner").value)

for sentence in corpus.test:
    model.predict(sentence,all_tag_prob=True)
    for token in sentence.tokens:
        pred.append(token.get_tag("ner").value)


print(corpus)
print(evaluate(real,pred))

