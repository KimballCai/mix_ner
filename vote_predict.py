import os
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import ELMoEmbeddings
from flair.embeddings import BertEmbeddings
from flair.embeddings import XLNetEmbeddings
from flair.embeddings import FlairEmbeddings
from flair.models import SequenceTagger
from eval.conlleval import evaluate
import numpy as np

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
str(real)

labels = tag_dictionary.get_items()
print(labels)

bert_read = pd.read_csv('result/tag/bert_256.csv')
bert_vote_list = bert_read['0'].values.tolist()
bert128_read = pd.read_csv('result/tag/bert_128.csv')
bert128_vote_list = bert128_read['0'].values.tolist()
bert64_read = pd.read_csv('result/tag/bert_64.csv')
bert64_vote_list = bert64_read['0'].values.tolist()
bert512_read = pd.read_csv('result/tag/bert_512.csv')
bert512_vote_list = bert512_read['0'].values.tolist()

elmo_read = pd.read_csv('result/tag/elmo_256.csv')
elmo_vote_list = elmo_read['0'].values.tolist()
elmo128_read = pd.read_csv('result/tag/elmo_128.csv')
elmo128_vote_list = elmo128_read['0'].values.tolist()
elmo64_read = pd.read_csv('result/tag/elmo_64.csv')
elmo64_vote_list = elmo64_read['0'].values.tolist()
elmo512_read = pd.read_csv('result/tag/elmo_512.csv')
elmo512_vote_list = elmo512_read['0'].values.tolist()

xlnet_read = pd.read_csv('result/tag/xlnet_256.csv')
xlnet_vote_list = xlnet_read['0'].values.tolist()
xlnet128_read = pd.read_csv('result/tag/xlnet_128.csv')
xlnet128_vote_list = xlnet128_read['0'].values.tolist()
xlnet64_read = pd.read_csv('result/tag/xlnet_64.csv')
xlnet64_vote_list = xlnet64_read['0'].values.tolist()
xlnet512_read = pd.read_csv('result/tag/xlnet_512.csv')
xlnet512_vote_list = xlnet512_read['0'].values.tolist()


flair_read = pd.read_csv('result/tag/flair_256.csv')
flair_vote_list = flair_read['0'].values.tolist()
flair128_read = pd.read_csv('result/tag/flair_128.csv')
flair128_vote_list = flair128_read['0'].values.tolist()
flair64_read = pd.read_csv('result/tag/flair_64.csv')
flair64_vote_list = flair_read['0'].values.tolist()
flair512_read = pd.read_csv('result/tag/flair_512.csv')
flair512_vote_list = flair512_read['0'].values.tolist()

b_e_read = pd.read_csv('result/tag/b_e.csv')
be_vote_list = b_e_read['0'].values.tolist()
b_x_read = pd.read_csv('result/tag/b_x.csv')
bx_vote_list = b_x_read['0'].values.tolist()
e_x_read = pd.read_csv('result/tag/e_x.csv')
ex_vote_list = e_x_read['0'].values.tolist()
x_f_read = pd.read_csv('result/tag/x_f.csv')
xf_vote_list = x_f_read['0'].values.tolist()
e_f_read = pd.read_csv('result/tag/e_f.csv')
ef_vote_list = e_f_read['0'].values.tolist()
b_f_read = pd.read_csv('result/tag/b_f.csv')
bf_vote_list = b_f_read['0'].values.tolist()

e_b_x_read = pd.read_csv('result/tag/e_b_x.csv')
ebx_vote_list = e_b_x_read['0'].values.tolist()
x_f_e_read = pd.read_csv('result/tag/x_f_e.csv')
xfe_vote_list = x_f_e_read['0'].values.tolist()
b_f_x_read = pd.read_csv('result/tag/b_f_x.csv')
bfx_vote_list = b_f_x_read['0'].values.tolist()
b_e_f_read = pd.read_csv('result/tag/b_e_f.csv')
bef_vote_list = b_e_f_read['0'].values.tolist()

x_f_e_b_read = pd.read_csv('result/tag/x_f_e_b.csv')
xfeb_vote_list = x_f_e_b_read['0'].values.tolist()

print("Already read all required file!")

print("--------------BERT EVALUATION---------------")
print(evaluate(real, bert_vote_list))
print("-------------------END----------------------")

print("--------------ELMO EVALUATION---------------")
print(evaluate(real, elmo_vote_list))
print("-------------------END----------------------")

print("--------------XLNET EVALUATION---------------")
print(evaluate(real, xlnet_vote_list))
print("-------------------END----------------------")

print("--------------POOL-FLAIR EVALUATION---------------")
# print(evaluate(real, flair_f_vote_list))
print(evaluate(real, flair_vote_list))
print("-------------------END----------------------")

print("--------------BE EVALUATION---------------")
print(evaluate(real, be_vote_list))
print("-------------------END----------------------")
print("--------------BX EVALUATION---------------")
print(evaluate(real, bx_vote_list))
print("-------------------END----------------------")
print("--------------EX EVALUATION---------------")
print(evaluate(real, ex_vote_list))
print("-------------------END----------------------")
print("--------------XF EVALUATION---------------")
print(evaluate(real, xf_vote_list))
print("-------------------END----------------------")
print("--------------EF EVALUATION---------------")
print(evaluate(real, ef_vote_list))
print("-------------------END----------------------")
print("--------------BF EVALUATION---------------")
print(evaluate(real, bf_vote_list))
print("-------------------END----------------------")

print("--------------EBX EVALUATION---------------")
print(evaluate(real, ebx_vote_list))
print("-------------------END----------------------")
print("--------------XFE EVALUATION---------------")
print(evaluate(real, xfe_vote_list))
print("-------------------END----------------------")
print("--------------BFX EVALUATION---------------")
print(evaluate(real, bfx_vote_list))
print("-------------------END----------------------")
print("--------------BEF VALUATION---------------")
print(evaluate(real, bef_vote_list))
print("-------------------END----------------------")

print("--------------XFEB EVALUATION---------------")
print(evaluate(real, xfeb_vote_list))
print("-------------------END----------------------")


# print("--------------FLAIR-BACKWARD EVALUATION---------------")
# print(evaluate(real, flair_b_vote_list))
# print("-------------------END----------------------")

def get_position_tag(tag_str):
    dash_index = tag_str.find("-")
    if dash_index == -1:
        return tag_str
    return tag_str[0:dash_index]


def get_classification_tag(tag_str):
    dash_index = tag_str.find("-")
    if dash_index == -1:
        return tag_str
    return tag_str[dash_index + 1:]

def vote(vote_list):
  vote = {'O': 0, 'B-ORG': 0, 'B-PER': 0, 'B-MISC': 0, 'B-LOC': 0, 'I-ORG': 0, 'I-PER': 0, 'I-MISC': 0, 'I-LOC': 0}
  for vote_tag in vote_list:
    vote[vote_tag] = vote[vote_tag] + 1
  final_tag = max(vote, key=vote.get)
  return final_tag

def hierarchy_vote(vote_list):
    '''
    vote_list=['B-LOC','B-MISC','I-LOC',...]
    '''
    classificatin_vote = {'ORG': 0, 'PER': 0, 'MISC': 0, 'LOC': 0, 'O': 0}
    position_vote = {'B': 0, 'I': 0, 'O': 0}
    for vote_tag in vote_list:
        temp_position = get_position_tag(vote_tag)
        position_vote[temp_position] = position_vote[temp_position] + 1

        temp_classification = get_classification_tag(vote_tag)
        classificatin_vote[temp_classification] = classificatin_vote[temp_classification] + 1

    final_position = max(position_vote, key=position_vote.get)
    final_classificatin = max(classificatin_vote, key=classificatin_vote.get)
    if (final_position == 'O') and (final_classificatin == 'O'):
        return 'O'
    return final_position + "-" + final_classificatin


############################ 3 Model #############################
vote_lists = []
# 确认几个ner predict list长度相等
for i in range(len(elmo_vote_list)):
    vote_lists.append([elmo_vote_list[i], xlnet_vote_list[i], flair_vote_list[i]])
    # vote_lists.append([bert_vote_list[i], elmo_vote_list[i]])
    # [[O,O,O],[B-LOC,O,O],...]

h_vote_predict = []
for vote_list in vote_lists:
  h_vote_predict.append(hierarchy_vote(vote_list))
  
print("--------------ELMO+XLNET+FLAIR HIERARCHY VOTE EVALUATION---------------")
print(evaluate(real,h_vote_predict))
print("-------------------ELMO+XLNET+FLAIR END----------------------")

vote_predict = []
for vote_list in vote_lists:
  vote_predict.append(vote(vote_list))

print("--------------ELMO+XLNET+FLAIR VOTE EVALUATION---------------")
print(evaluate(real,vote_predict))
print("-------------------ELMO+XLNET+FLAIR END----------------------")
############################ 3 Model #############################

############################ 3 Model #############################
vote_lists = []
# 确认几个ner predict list长度相等
for i in range(len(elmo_vote_list)):
    vote_lists.append([bert_vote_list[i], xlnet_vote_list[i], flair_vote_list[i]])
    # vote_lists.append([bert_vote_list[i], elmo_vote_list[i]])
    # [[O,O,O],[B-LOC,O,O],...]

h_vote_predict = []
for vote_list in vote_lists:
  h_vote_predict.append(hierarchy_vote(vote_list))
  
print("--------------BERT+XLNET+FLAIR HIERARCHY VOTE EVALUATION---------------")
print(evaluate(real,h_vote_predict))
print("-------------------BERT+XLNET+FLAIR END----------------------")

vote_predict = []
for vote_list in vote_lists:
  vote_predict.append(vote(vote_list))

print("--------------BERT+XLNET+FLAIR VOTE EVALUATION---------------")
print(evaluate(real,vote_predict))
print("-------------------BERT+XLNET+FLAIR END----------------------")
############################ 3 Model #############################

############################ 3 Model #############################
vote_lists = []
# 确认几个ner predict list长度相等
for i in range(len(elmo_vote_list)):
    vote_lists.append([bert_vote_list[i], elmo_vote_list[i], flair_vote_list[i]])
    # vote_lists.append([bert_vote_list[i], elmo_vote_list[i]])
    # [[O,O,O],[B-LOC,O,O],...]

h_vote_predict = []
for vote_list in vote_lists:
  h_vote_predict.append(hierarchy_vote(vote_list))
  
print("--------------BERT+ELMO+FLAIR HIERARCHY VOTE EVALUATION---------------")
print(evaluate(real,h_vote_predict))
print("-------------------BERT+ELMO+FLAIR END----------------------")

vote_predict = []
for vote_list in vote_lists:
  vote_predict.append(vote(vote_list))

print("--------------BERT+ELMO+FLAIR VOTE EVALUATION---------------")
print(evaluate(real,vote_predict))
print("-------------------BERT+ELMO+FLAIR END----------------------")
############################ 3 Model #############################

############################ 3 Model #############################
vote_lists = []
# 确认几个ner predict list长度相等
for i in range(len(elmo_vote_list)):
    vote_lists.append([bert_vote_list[i], elmo_vote_list[i], xlnet_vote_list[i]])
    # vote_lists.append([bert_vote_list[i], elmo_vote_list[i]])
    # [[O,O,O],[B-LOC,O,O],...]

h_vote_predict = []
for vote_list in vote_lists:
  h_vote_predict.append(hierarchy_vote(vote_list))
  
print("--------------BERT+ELMO+XLNET HIERARCHY VOTE EVALUATION---------------")
print(evaluate(real,h_vote_predict))
print("-------------------BERT+ELMO+XLNET END----------------------")

vote_predict = []
for vote_list in vote_lists:
  vote_predict.append(vote(vote_list))

print("--------------BERT+ELMO+XLNET VOTE EVALUATION---------------")
print(evaluate(real,vote_predict))
print("-------------------BERT+ELMO+XLNET END----------------------")
############################ 3 Model #############################

############################ 4 Model #############################
vote_lists = []
# 确认几个ner predict list长度相等
for i in range(len(elmo_vote_list)):
    vote_lists.append([bert_vote_list[i], elmo_vote_list[i], xlnet_vote_list[i], flair_vote_list[i]])
    # vote_lists.append([bert_vote_list[i], elmo_vote_list[i]])
    # [[O,O,O],[B-LOC,O,O],...]

h_vote_predict = []
for vote_list in vote_lists:
  h_vote_predict.append(hierarchy_vote(vote_list))
  
print("--------------BERT+ELMO+XLNET+FLAIR HIERARCHY VOTE EVALUATION---------------")
print(evaluate(real,h_vote_predict))
print("-------------------BERT+ELMO+XLNET+FLAIR END----------------------")

vote_predict = []
for vote_list in vote_lists:
  vote_predict.append(vote(vote_list))

print("--------------BERT+ELMO+XLNET+FLAIR VOTE EVALUATION---------------")
print(evaluate(real,vote_predict))
print("-------------------BERT+ELMO+XLNET+FLAIR END----------------------")
############################ 4 Model #############################

############################ 8 Model #############################
vote_lists = []
# 确认几个ner predict list长度相等
for i in range(len(elmo_vote_list)):
    vote_lists.append([bert_vote_list[i],bert128_vote_list[i] , elmo_vote_list[i], elmo128_vote_list[i], xlnet_vote_list[i], xlnet128_vote_list[i], flair_vote_list[i], flair128_vote_list[i]])

h_vote_predict = []
for vote_list in vote_lists:
  h_vote_predict.append(hierarchy_vote(vote_list))
  
print("--------------MULTI 8 MODEL (128+256) HIERARCHY VOTE EVALUATION---------------")
print(evaluate(real,h_vote_predict))
print("-------------------MULTI 8 MODEL (128+256) END----------------------")

vote_predict = []
for vote_list in vote_lists:
  vote_predict.append(vote(vote_list))

print("--------------MULTI 8 MODEL (128+256) VOTE EVALUATION---------------")
print(evaluate(real,vote_predict))
print("-------------------MULTI 8 MODEL (128+256) END----------------------")
############################ 8 Model #############################

############################ 16 Model #############################
vote_lists = []
# 确认几个ner predict list长度相等
for i in range(len(elmo_vote_list)):
    vote_lists.append([bert_vote_list[i],bert128_vote_list[i] ,bert64_vote_list[i],bert512_vote_list[i], elmo_vote_list[i], elmo128_vote_list[i],elmo64_vote_list[i],elmo512_vote_list[i], xlnet_vote_list[i], xlnet128_vote_list[i],xlnet64_vote_list[i],xlnet512_vote_list[i], flair_vote_list[i], flair128_vote_list[i],flair64_vote_list[i],flair512_vote_list[i]])
    # vote_lists.append([bert_vote_list[i], elmo_vote_list[i]])
    # [[O,O,O],[B-LOC,O,O],...]

h_vote_predict = []
for vote_list in vote_lists:
  h_vote_predict.append(hierarchy_vote(vote_list))
  
print("--------------MULTI 16 MODEL (64+128+256+512) HIERARCHY VOTE EVALUATION---------------")
print(evaluate(real,h_vote_predict))
print("-------------------MULTI 16 MODEL (64+128+256+512) END----------------------")

vote_predict = []
for vote_list in vote_lists:
  vote_predict.append(vote(vote_list))

print("--------------MULTI 16 MODEL (64+128+256+512) VOTE EVALUATION---------------")
print(evaluate(real,vote_predict))
print("-------------------MULTI 16 MODEL (64+128+256+512)) END----------------------")
############################ 16 Model #############################

############################ BEST 8 Model #############################
vote_lists = []
# 确认几个ner predict list长度相等
for i in range(len(elmo_vote_list)):
    vote_lists.append([bert_vote_list[i],bert128_vote_list[i] ,bert64_vote_list[i], elmo_vote_list[i], elmo128_vote_list[i],elmo64_vote_list[i],elmo512_vote_list[i], flair_vote_list[i]])
    # vote_lists.append([bert_vote_list[i], elmo_vote_list[i]])
    # [[O,O,O],[B-LOC,O,O],...]

h_vote_predict = []
for vote_list in vote_lists:
  h_vote_predict.append(hierarchy_vote(vote_list))
  
print("--------------1st BEST 8 HIERARCHY VOTE EVALUATION---------------")
print(evaluate(real,h_vote_predict))
print("-------------------1st BEST 8 END----------------------")

vote_predict = []
for vote_list in vote_lists:
  vote_predict.append(vote(vote_list))

print("--------------1st BEST 8 VOTE EVALUATION---------------")
print(evaluate(real,vote_predict))
print("-------------------1st BEST 8 END----------------------")
############################ BEST 8 Model #############################

############################ BEST 8 Model #############################
vote_lists = []
# 确认几个ner predict list长度相等
for i in range(len(elmo_vote_list)):
    vote_lists.append([bert_vote_list[i],bert128_vote_list[i] ,bert64_vote_list[i], elmo_vote_list[i], elmo128_vote_list[i],elmo64_vote_list[i],elmo512_vote_list[i], xlnet_vote_list[i]])
    # vote_lists.append([bert_vote_list[i], elmo_vote_list[i]])
    # [[O,O,O],[B-LOC,O,O],...]

h_vote_predict = []
for vote_list in vote_lists:
  h_vote_predict.append(hierarchy_vote(vote_list))
  
print("--------------2nd BEST 8 HIERARCHY VOTE EVALUATION---------------")
print(evaluate(real,h_vote_predict))
print("-------------------2nd BEST 8 END----------------------")

vote_predict = []
for vote_list in vote_lists:
  vote_predict.append(vote(vote_list))

print("--------------2nd BEST 8 VOTE EVALUATION---------------")
print(evaluate(real,vote_predict))
print("-------------------2nd BEST 8 END----------------------")
############################ BEST 8 Model #############################