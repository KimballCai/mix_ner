#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''
@version:0.1
@author:Cai Qingpeng
@file: train.py
@time: 2020/3/18 8:21 PM
'''

import os
from datetime import datetime
import argparse

from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import ELMoEmbeddings
from flair.embeddings import BertEmbeddings
from flair.embeddings import XLNetEmbeddings


os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"


def parse_args():
    # parse arguments
    ## general
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--embed', default='bert',
                            help='elmo bert flair')
    return arg_parser.parse_args()

ARGS = parse_args()

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
embed = ARGS.embed
print(embed)
if embed == "elmo":
    embedding = ELMoEmbeddings("small")
elif embed == "bert":
    embedding = BertEmbeddings()
elif embed == "xlnet":
    embedding = XLNetEmbeddings()

# 5. initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embedding,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)

# 6. initialize trainer
from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

# 7. start training
trainer.train("./log/%s_%s/" % (ARGS.embed, str(timestamp)),
              learning_rate=0.01,
              mini_batch_size=64,
              max_epochs=150)


# from flair.visual.training_curves import Plotter
# plotter = Plotter()
# plotter.plot_weights('./log/elmo/weights.txt')


