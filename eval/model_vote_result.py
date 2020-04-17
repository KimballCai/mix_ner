from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger

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

# =========256========
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

def get_vote_tag_list(model):
  predict_tag_list = []
  for sentence in corpus.test:
      model.predict(sentence,all_tag_prob=True)
      for token in sentence.tokens:
          predict_tag_list.append(token.get_tag("ner").value)
  return predict_tag_list

bert_64_vote_list = get_vote_tag_list(bert_64_model)
print("Get bert 64 result")
bert_128_vote_list = get_vote_tag_list(bert_128_model)
print("Get bert 128 result")
bert_512_vote_list = get_vote_tag_list(bert_512_model)
print("Get bert 512 result")
# elmo_64_vote_list = get_vote_tag_list(elmo_64_model)
# print("Get elmo 64 result")
# elmo_128_vote_list = get_vote_tag_list(elmo_128_model)
# print("Get elmo 128 result")
# elmo_512_vote_list = get_vote_tag_list(elmo_512_model)
# print("Get elmo 512 result")
# xlnet_64_vote_list = get_vote_tag_list(xlnet_64_model)
# print("Get xlnet 64 result")
# xlnet_128_vote_list = get_vote_tag_list(xlnet_128_model)
# print("Get xlnet 128 result")
# xlnet_512_vote_list = get_vote_tag_list(xlnet_512_model)
# print("Get xlnet 512 result")
# flair_64_vote_list = get_vote_tag_list(flair_64_model)
# print("Get flair 64 result")
# flair_128_vote_list = get_vote_tag_list(flair_128_model)
# print("Get flair 128 result")
# flair_512_vote_list = get_vote_tag_list(flair_512_model)
# print("Get flair 512 result")

def clean_prediction(p_list):
  return_list = p_list
  check_set = {'O','B-ORG','B-MISC','B-PER','I-PER','B-LOC','I-ORG','I-MISC','I-LOC'}
  for i in range(len(return_list)):
    if not return_list[i] in check_set:
      return_list[i] = 'O'
  return return_list

bert_64_vote_list = clean_prediction(bert_64_vote_list)
bert_128_vote_list = clean_prediction(bert_128_vote_list)
bert_512_vote_list = clean_prediction(bert_512_vote_list)
# elmo_64_vote_list = clean_prediction(elmo_64_vote_list)
# elmo_128_vote_list = clean_prediction(elmo_128_vote_list)
# elmo_512_vote_list = clean_prediction(elmo_512_vote_list)
# xlnet_64_vote_list = clean_prediction(xlnet_64_vote_list)
# xlnet_128_vote_list = clean_prediction(xlnet_128_vote_list)
# xlnet_512_vote_list = clean_prediction(xlnet_512_vote_list)
# flair_64_vote_list = clean_prediction(flair_64_vote_list)
# flair_128_vote_list = clean_prediction(flair_128_vote_list)
# flair_512_vote_list = clean_prediction(flair_512_vote_list)

import pandas as pd
bert_64_result_list = pd.DataFrame(bert_64_vote_list)
bert_64_result_list.to_csv('result/tag/bert_64.csv')
print("Saving bert 64 result done")
bert_128_result_list = pd.DataFrame(bert_128_vote_list)
bert_128_result_list.to_csv('result/tag/bert_128.csv')
print("Saving bert 128 result done")
bert_512_result_list = pd.DataFrame(bert_512_vote_list)
bert_512_result_list.to_csv('result/tag/bert_512.csv')
print("Saving bert 512 result done")

# elmo_64_result_list = pd.DataFrame(elmo_64_vote_list)
# elmo_64_result_list.to_csv('result/tag/elmo_64.csv')
# print("Saving elmo 64 result done")
# elmo_128_result_list = pd.DataFrame(elmo_128_vote_list)
# elmo_128_result_list.to_csv('result/tag/elmo_128.csv')
# print("Saving elmo 128 result done")
# elmo_512_result_list = pd.DataFrame(elmo_512_vote_list)
# elmo_512_result_list.to_csv('result/tag/elmo_512.csv')
# print("Saving elmo 512 result done")

# xlnet_64_result_list = pd.DataFrame(xlnet_64_vote_list)
# xlnet_64_result_list.to_csv('result/tag/xlnet_64.csv')
# print("Saving xlnet 64 result done")
# xlnet_128_result_list = pd.DataFrame(xlnet_128_vote_list)
# xlnet_128_result_list.to_csv('result/tag/xlnet_128.csv')
# print("Saving xlnet 128 result done")
# xlnet_512_result_list = pd.DataFrame(xlnet_512_vote_list)
# xlnet_512_result_list.to_csv('result/tag/xlnet_512.csv')
# print("Saving xlnet 512 result done")

# flair_64_result_list = pd.DataFrame(flair_64_vote_list)
# flair_64_result_list.to_csv('result/tag/flair_64.csv')
# print("Saving flair 64 result done")
# flair_128_result_list = pd.DataFrame(flair_128_vote_list)
# flair_128_result_list.to_csv('result/tag/flair_128.csv')
# print("Saving flair 128 result done")
# flair_512_result_list = pd.DataFrame(flair_512_vote_list)
# flair_512_result_list.to_csv('result/tag/flair_512.csv')
# print("Saving flair 512 result done")
