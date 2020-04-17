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

# save_result("mix_be_20200408202049_256", "b_e")
# save_result("mix_bf_20200408202153_256", "b_f")
# save_result("mix_bx_20200408202350_256", "b_x")
save_result("mix_xf_20200401202715", "x_f")
# save_result("mix_ex_20200408202816_256", "e_x")
save_result("mix_ef_20200408202744_256", "e_f")
save_result("mix_bef_20200409142925_256", "b_e_f")
# save_result("mix_ebx_20200324145746", "e_b_x")
save_result("mix_xfe_20200401204553_256", "x_f_e")
save_result("mix_bfx_20200409143007_256", "b_f_x")
save_result("mix_xfeb_20200401210102_256", "x_f_e_b")


# # =========256========
# be_model = SequenceTagger.load('./log/mix_be_20200408202049_256/best-model.pt')
# bf_model = SequenceTagger.load('./log/mix_bf_20200408202153_256/best-model.pt')
# bx_model = SequenceTagger.load('./log/mix_bx_20200408202350_256/best-model.pt')
# xf_model = SequenceTagger.load('./log/mix_xf_20200401202715/best-model.pt')
# ex_model = SequenceTagger.load('./log/mix_ex_20200408202816_256/best-model.pt')
# ef_model = SequenceTagger.load('./log/mix_ef_20200408202744_256/best-model.pt')
#
# bef_model = SequenceTagger.load('./log/mix_bef_20200409142925_256/best-model.pt')
# ebx_model = SequenceTagger.load('./log/mix_ebx_20200324145746/best-model.pt')
# xfe_model = SequenceTagger.load('./log/mix_xfe_20200401204553_256/best-model.pt')
# bfx_model = SequenceTagger.load('./log/mix_bfx_20200409143007_256/best-model.pt')
#
# xfeb_model = SequenceTagger.load('./log/mix_xfeb_20200401210102_256/best-model.pt')
#
#
# be_vote_list = get_vote_tag_list(be_model)
# print("Get be result")
# be_result_list = pd.DataFrame(be_vote_list)
# be_result_list.to_csv('result/tag/b_e.csv')
# print("Saving be result done")
# bf_vote_list = get_vote_tag_list(bf_model)
# print("Get bf result")
# bf_result_list = pd.DataFrame(bf_vote_list)
# bf_result_list.to_csv('result/tag/b_f.csv')
# print("Saving bf result done")
# bx_vote_list = get_vote_tag_list(bx_model)
# print("Get bx result")
# bx_result_list = pd.DataFrame(bx_vote_list)
# bx_result_list.to_csv('result/tag/b_x.csv')
# print("Saving bx result done")
# xf_vote_list = get_vote_tag_list(xf_model)
# print("Get xf result")
# xf_result_list = pd.DataFrame(xf_vote_list)
# xf_result_list.to_csv('result/tag/x_f.csv')
# print("Saving xf result done")
# ex_vote_list = get_vote_tag_list(ex_model)
# print("Get ex result")
# ex_result_list = pd.DataFrame(ex_vote_list)
# ex_result_list.to_csv('result/tag/e_x.csv')
# print("Saving ex result done")
# ef_vote_list = get_vote_tag_list(ef_model)
# print("Get ef result")
# ef_result_list = pd.DataFrame(ef_vote_list)
# ef_result_list.to_csv('result/tag/e_f.csv')
# print("Saving ef result done")
#
# bef_vote_list = get_vote_tag_list(bef_model)
# print("Get bef result")
# bef_result_list = pd.DataFrame(bef_vote_list)
# bef_result_list.to_csv('result/tag/b_e_f.csv')
# print("Saving bef result done")
# ebx_vote_list = get_vote_tag_list(ebx_model)
# print("Get ebx result")
# ebx_result_list = pd.DataFrame(ebx_vote_list)
# ebx_result_list.to_csv('result/tag/e_b_x.csv')
# print("Saving ebx result done")
# xfe_vote_list = get_vote_tag_list(xfe_model)
# print("Get xfe result")
# xfe_result_list = pd.DataFrame(xfe_vote_list)
# xfe_result_list.to_csv('result/tag/x_f_e.csv')
# print("Saving xfe result done")
# bfx_vote_list = get_vote_tag_list(bfx_model)
# print("Get bfx result")
# bfx_result_list = pd.DataFrame(bfx_vote_list)
# bfx_result_list.to_csv('result/tag/b_f_x.csv')
# print("Saving bfx result done")
#
# xfeb_vote_list = get_vote_tag_list(xfeb_model)
# print("Get xfeb result")
# xfeb_result_list = pd.DataFrame(xfeb_vote_list)
# xfeb_result_list.to_csv('result/tag/x_f_e_b.csv')
# print("Saving xfeb result done")
