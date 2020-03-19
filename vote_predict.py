import os
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import ELMoEmbeddings
from flair.embeddings import BertEmbeddings
from flair.embeddings import XLNetEmbeddings
from flair.models import SequenceTagger
from conlleval import evaluate
import numpy as np


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

bert_model = SequenceTagger.load('./log/bert_20200319181345/final-model.pt')
elmo_model = SequenceTagger.load('./log/elmo_20200319142948/best-model.pt')
xlnet_model = SequenceTagger.load('./log/xlnet_20200319155116/best-model.pt')

def get_vote_tag_list(model):
  predict_tag_list = []
  for sentence in corpus.test:
      model.predict(sentence,all_tag_prob=True)
      for token in sentence.tokens:
          predict_tag_list.append(token.get_tag("ner").value)
  return predict_tag_list

bert_vote_list = get_vote_tag_list(bert_model)
elmo_vote_list = get_vote_tag_list(elmo_model)
xlnet_vote_list = get_vote_tag_list(xlnet_model)


def clean_prediction(p_list):
  return_list = p_list
  check_set = {'O','B-ORG','B-MISC','B-PER','I-PER','B-LOC','I-ORG','I-MISC','I-LOC'}
  for i in range(len(return_list)):
    if not return_list[i] in check_set:
      return_list[i] = 'O'
      print("There is a error check: ",tag," in index of ",i)
  return return_list

bert_vote_list = clean_prediction(bert_vote_list)
elmo_vote_list = clean_prediction(elmo_vote_list)
xlnet_vote_list = clean_prediction(xlnet_vote_list)


print("--------------BERT EVALUATION---------------")
print(evaluate(real, bert_vote_list))
print("-------------------END----------------------")

print("--------------ELMO EVALUATION---------------")
print(evaluate(real, elmo_vote_list))
print("-------------------END----------------------")

print("--------------XLNET EVALUATION---------------")
print(evaluate(real, xlnet_vote_list))
print("-------------------END----------------------")


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

vote_lists = []

# 确认几个ner predict list长度相等
for i in range(len(elmo_vote_list)):
    vote_lists.append([bert_vote_list[i], elmo_vote_list[i], xlnet_vote_list[i]])
    # vote_lists.append([bert_vote_list[i], elmo_vote_list[i]])
    # [[O,O,O],[B-LOC,O,O],...]

vote_predict = []
for vote_list in vote_lists:
    vote_predict.append(vote(vote_list))


print("--------------XLNET EVALUATION---------------")
print(evaluate(real,vote_predict))
print("-------------------END----------------------")