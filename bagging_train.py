#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''
@version:0.1
@author:Cai Qingpeng
@file: bagging_train.py
@time: 2020/4/1 10:00 PM
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from flair.data import Corpus
from flair.datasets import ColumnCorpus

columns = {0: 'text', 1: '_', 2: '_', 3: 'ner'}

data_folder = './data'

corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file='train.txt',
                              test_file='test.txt',
                              dev_file='valid.txt')
print(corpus)
tag_type = 'ner'
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary)

from flair.embeddings import ELMoEmbeddings,BertEmbeddings,FlairEmbeddings
from flair.models import SequenceTagger

elmo_tagger = SequenceTagger(hidden_size=256,
                             embeddings=ELMoEmbeddings('small'),
                             tag_dictionary=tag_dictionary,
                             tag_type=tag_type,
                             use_crf=True)

bert_tagger = SequenceTagger(hidden_size=256,
                             embeddings=BertEmbeddings(),
                             tag_dictionary=tag_dictionary,
                             tag_type=tag_type,
                             use_crf=True)
# flair_tagger = SequenceTagger(hidden_size=256,
#                               embeddings=FlairEmbeddings('en-forward'),
#                               tag_dictionary=tag_dictionary,
#                               tag_type=tag_type,
#                               use_crf=True)

from ensemble_trainer import ModelStackTrainer
from datetime import datetime



trainer = ModelStackTrainer([bert_tagger,elmo_tagger], corpus)

timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

# 7. start training
trainer.train("./log/%s_%s/" % ("avg_loss", str(timestamp)),
              learning_rate=0.01,
              mini_batch_size=32,
              max_epochs=150)
