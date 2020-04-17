import os
import numpy as np
from eval.conlleval import evaluate
from flair.data import Corpus
from flair.datasets import ColumnCorpus
import pandas as pd

THRESHHOLD = 0.3

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

bert_read = pd.read_csv('result/prob/bert_256.csv')
bert_poss = bert_read['0'].values.tolist()
bert128_read = pd.read_csv('result/prob/bert_128.csv')
bert128_poss = bert128_read['0'].values.tolist()
bert64_read = pd.read_csv('result/prob/bert_64.csv')
bert64_poss = bert64_read['0'].values.tolist()
bert512_read = pd.read_csv('result/prob/bert_512.csv')
bert512_poss = bert512_read['0'].values.tolist()

elmo_read = pd.read_csv('result/prob/elmo_256.csv')
elmo_poss = elmo_read['0'].values.tolist()
elmo128_read = pd.read_csv('result/prob/elmo_128.csv')
elmo128_poss = elmo128_read['0'].values.tolist()
elmo64_read = pd.read_csv('result/prob/elmo_64.csv')
elmo64_poss = elmo64_read['0'].values.tolist()
elmo512_read = pd.read_csv('result/prob/elmo_512.csv')
elmo512_poss = elmo512_read['0'].values.tolist()

xlnet_read = pd.read_csv('result/prob/xlnet_256.csv')
xlnet_poss = xlnet_read['0'].values.tolist()
xlnet128_read = pd.read_csv('result/prob/xlnet_128.csv')
xlnet128_poss = xlnet128_read['0'].values.tolist()
xlnet64_read = pd.read_csv('result/prob/xlnet_64.csv')
xlnet64_poss = xlnet64_read['0'].values.tolist()
xlnet512_read = pd.read_csv('result/prob/xlnet_512.csv')
xlnet512_poss = xlnet512_read['0'].values.tolist()

flair_read = pd.read_csv('result/prob/flair_256.csv')
flair_poss = flair_read['0'].values.tolist()
flair128_read = pd.read_csv('result/prob/flair_128.csv')
flair128_poss = flair128_read['0'].values.tolist()
flair64_read = pd.read_csv('result/prob/flair_64.csv')
flair64_poss = flair64_read['0'].values.tolist()
flair512_read = pd.read_csv('result/prob/flair_512.csv')
flair512_poss = flair512_read['0'].values.tolist()

print("Already read all required file!")

def all_zero(one_word_distribution):
  for i in one_word_distribution:
    if i != 0.0:
      return False
  return True

def one_hot(poss_candidate):
  one_hot_list = []
  for each_poss in poss_candidate:
    # index_max = each_poss.index(max(each_poss))
    index_max = np.argmax(each_poss)
    each_onehot = [0] * 12
    if (each_poss[index_max]>=THRESHHOLD):
      each_onehot[index_max] = 1
    one_hot_list.append(each_onehot)
  return one_hot_list

def confidence_vote(poss_candidate):
  '''
  input: poss_span=[[0.1, 0.2, ...] , [0.1, 0.2, ...], ...]
  output: label = ['O','B-LOC',...]
  '''
  # 转成one-hot: [[0,0,1,..],[],]
  one_hot_candidata = one_hot(poss_candidate)
  
  gather_vote_distribution = [0]*12
  candidate_num = len(one_hot_candidata)
  for i in range(candidate_num):
    for j in range(12):
      gather_vote_distribution[j] += one_hot_candidata[i][j]

  # 判断能不能用confidence_vote, 如果全是0就用相加用最大值
  if all_zero(gather_vote_distribution):
    print("全是0")
    sum_poss = [0]*12
    for i in range(len(poss_candidate)):
      for j in range(12):
        sum_poss[j]+=poss_candidate[i][j]
    vote_index = sum_poss.index(max(sum_poss))
    return tag_dictionary.get_item_for_index(vote_index)

  # 如果可以用就取最大值
  vote_index = gather_vote_distribution.index(max(gather_vote_distribution))
  return tag_dictionary.get_item_for_index(vote_index)

bert_poss_list = np.reshape(bert_poss,(-1,len(labels)))
bert128_poss_list = np.reshape(bert128_poss,(-1,len(labels)))
bert64_poss_list = np.reshape(bert64_poss,(-1,len(labels)))
bert512_poss_list = np.reshape(bert512_poss,(-1,len(labels)))

elmo_poss_list = np.reshape(elmo_poss,(-1,len(labels)))
elmo128_poss_list = np.reshape(elmo128_poss,(-1,len(labels)))
elmo64_poss_list = np.reshape(elmo64_poss,(-1,len(labels)))
elmo512_poss_list = np.reshape(elmo512_poss,(-1,len(labels)))

flair_poss_list = np.reshape(flair_poss,(-1,len(labels)))
flair128_poss_list = np.reshape(flair128_poss,(-1,len(labels)))
flair64_poss_list = np.reshape(flair64_poss,(-1,len(labels)))
flair512_poss_list = np.reshape(flair512_poss,(-1,len(labels)))

xlnet_poss_list = np.reshape(xlnet_poss,(-1,len(labels)))
xlnet128_poss_list = np.reshape(xlnet128_poss,(-1,len(labels)))
xlnet64_poss_list = np.reshape(xlnet64_poss,(-1,len(labels)))
xlnet512_poss_list = np.reshape(xlnet512_poss,(-1,len(labels)))


#################BERT+ELMO+XLNET MODEL#########################
print("============================THREE MODEL:BERT+ELMO+XLNET=================================")
confidence_vote_lists = []
for i in range(len(bert_poss_list)):
  confidence_vote_lists.append([bert_poss_list[i],xlnet_poss_list[i],elmo_poss_list[i]])

confidence_predict = []
for confidence_vote_list in confidence_vote_lists:
  confidence_predict.append(confidence_vote(confidence_vote_list))

print(evaluate(real,confidence_predict))
print("============================THREE MODEL:BERT+ELMO+XLNET=================================")
#################BERT+ELMO+XLNET MODEL#########################

#################BERT+ELMO+FLAIR MODEL#########################
print("============================THREE MODEL:BERT+ELMO+FLAIR=================================")
confidence_vote_lists = []
for i in range(len(bert_poss_list)):
  confidence_vote_lists.append([bert_poss_list[i],flair_poss_list[i],elmo_poss_list[i]])

confidence_predict = []
for confidence_vote_list in confidence_vote_lists:
  confidence_predict.append(confidence_vote(confidence_vote_list))

print(evaluate(real,confidence_predict))
print("============================THREE MODEL:BERT+ELMO+FLAIR=================================")
#################BERT+ELMO+FLAIR MODEL#########################

#################BERT+XLNET+FLAIR MODEL#########################
print("============================THREE MODEL:BERT+XLNET+FLAIR=================================")
confidence_vote_lists = []
for i in range(len(bert_poss_list)):
  confidence_vote_lists.append([bert_poss_list[i],flair_poss_list[i],xlnet_poss_list[i]])

confidence_predict = []
for confidence_vote_list in confidence_vote_lists:
  confidence_predict.append(confidence_vote(confidence_vote_list))

print(evaluate(real,confidence_predict))
print("============================THREE MODEL:BERT+XLNET+FLAIR=================================")
#################BERT+XLNET+FLAIR MODEL#########################

#################ELMO+XLNET+FLAIR MODEL#########################
print("============================THREE MODEL:ELMO+XLNET+FLAIR=================================")
confidence_vote_lists = []
for i in range(len(bert_poss_list)):
  confidence_vote_lists.append([flair_poss_list[i],xlnet_poss_list[i],elmo_poss_list[i]])

confidence_predict = []
for confidence_vote_list in confidence_vote_lists:
  confidence_predict.append(confidence_vote(confidence_vote_list))

print(evaluate(real,confidence_predict))
print("============================THREE MODEL:ELMO+XLNET+FLAIR=================================")
#################ELMO+XLNET+FLAIR MODEL#########################

#################BERT+ELMO+XLNET+FLAIR MODEL#########################
print("============================FOUR MODEL:BERT+ELMO+XLNET+FLAIR=================================")
confidence_vote_lists = []
for i in range(len(bert_poss_list)):
  confidence_vote_lists.append([bert_poss_list[i],flair_poss_list[i],xlnet_poss_list[i],elmo_poss_list[i]])

confidence_predict = []
for confidence_vote_list in confidence_vote_lists:
  confidence_predict.append(confidence_vote(confidence_vote_list))

print(evaluate(real,confidence_predict))
#################BERT+ELMO+XLNET+FLAIR MODEL#########################

#################8 MODEL#########################
print("============================8 MODEL=================================")
confidence_vote_lists = []
for i in range(len(bert_poss_list)):
  confidence_vote_lists.append([bert_poss_list[i],bert128_poss_list[i],flair_poss_list[i],flair128_poss_list[i],xlnet_poss_list[i],xlnet128_poss_list[i],elmo_poss_list[i],elmo128_poss_list[i]])

confidence_predict = []
for confidence_vote_list in confidence_vote_lists:
  confidence_predict.append(confidence_vote(confidence_vote_list))

print(evaluate(real,confidence_predict))
#################8 MODEL#########################

#################16 MODEL#########################
print("============================16 MODEL=================================")
confidence_vote_lists = []
for i in range(len(bert_poss_list)):
  confidence_vote_lists.append([bert_poss_list[i],bert128_poss_list[i],bert64_poss_list[i],bert512_poss_list[i],flair_poss_list[i],flair128_poss_list[i],flair64_poss_list[i],flair512_poss_list[i],xlnet_poss_list[i],xlnet128_poss_list[i],xlnet64_poss_list[i],xlnet512_poss_list[i],elmo_poss_list[i],elmo128_poss_list[i],elmo64_poss_list[i],elmo512_poss_list[i]])

confidence_predict = []
for confidence_vote_list in confidence_vote_lists:
  confidence_predict.append(confidence_vote(confidence_vote_list))

print(evaluate(real,confidence_predict))
#################16 MODEL#########################

#################BEST 8 MODEL#########################
print("============================1st BEST 8 MODEL=================================")
confidence_vote_lists = []
for i in range(len(bert_poss_list)):
  confidence_vote_lists.append([bert_poss_list[i],bert128_poss_list[i],bert64_poss_list[i],flair_poss_list[i],elmo_poss_list[i],elmo128_poss_list[i],elmo64_poss_list[i],elmo512_poss_list[i]])

confidence_predict = []
for confidence_vote_list in confidence_vote_lists:
  confidence_predict.append(confidence_vote(confidence_vote_list))

print(evaluate(real,confidence_predict))
#################BEST 8 MODEL#########################

#################BEST 8 MODEL#########################
print("============================2nd BEST 8 MODEL=================================")
confidence_vote_lists = []
for i in range(len(bert_poss_list)):
  confidence_vote_lists.append([bert_poss_list[i],bert128_poss_list[i],bert64_poss_list[i],xlnet_poss_list[i],elmo_poss_list[i],elmo128_poss_list[i],elmo64_poss_list[i],elmo512_poss_list[i]])

confidence_predict = []
for confidence_vote_list in confidence_vote_lists:
  confidence_predict.append(confidence_vote(confidence_vote_list))

print(evaluate(real,confidence_predict))
#################BEST 8 MODEL#########################
