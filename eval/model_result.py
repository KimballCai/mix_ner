import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
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


def clean_prediction(p_list):
  return_list = p_list
  check_set = {'O','B-ORG','B-MISC','B-PER','I-PER','B-LOC','I-ORG','I-MISC','I-LOC'}
  for i in range(len(return_list)):
    if not return_list[i] in check_set:
      return_list[i] = 'O'
      # print("There is a error check: ",check_set," in index of ",i)
  return return_list

def get_vote_tag_list(model):
  predict_tag_list = []
  for sentence in corpus.test:
      model.predict(sentence,all_tag_prob=True)
      for token in sentence.tokens:
          predict_tag_list.append(token.get_tag("ner").value)
  return clean_prediction(predict_tag_list)

if not os.path.exists("./result/tag/"):
    os.mkdir("./result/")
    os.mkdir("./result/tag/")


def save_result(file_name, model_name):
    model = SequenceTagger.load('./log/%s/best-model.pt' % file_name)
    vote_list = get_vote_tag_list(model)
    print("Get %s result" % model_name)
    result_list = pd.DataFrame(vote_list)
    result_list.to_csv('result/tag/%s.csv' % model_name)
    print("Saving %s result done" % model_name)


save_result("mix_be_20200408202049_256", "b_e")
# save_result("mix_bf_20200408202153_256", "b_f")
# save_result("mix_bx_20200408202350_256", "b_x")
# save_result("mix_xf_20200401202715", "x_f")
# save_result("mix_ex_20200408202816_256", "e_x")
# save_result("mix_ef_20200408202744_256", "e_f")
# save_result("mix_bef_20200409142925_256", "b_e_f")
# save_result("mix_ebx_20200324145746", "e_b_x")
# save_result("mix_xfe_20200401204553_256", "x_f_e")
# save_result("mix_bfx_20200409143007_256", "b_f_x")
# save_result("mix_xfeb_20200401210102_256", "x_f_e_b")