import os
import numpy as np
from eval.conlleval import evaluate
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

#################BERT+ELMO MODEL#########################
print("============================TWO MODEL:BERT+ELMO=================================")
print("****** avg prediction ******")
avg_pred = get_mix_preds_by_result([bert_poss,elmo_poss],"avg")
print(evaluate(real,avg_pred))
print("****** confidence prediction ******")
confidence_pred = get_mix_preds_by_result([bert_poss,elmo_poss],"confidence")
print(evaluate(real,confidence_pred))
print("============================TWO MODEL:BERT+ELMO=================================")
#################BERT+ELMO MODEL#########################

#################BERT+XLNET MODEL#########################
print("============================TWO MODEL:BERT+XLNET=================================")
print("****** avg prediction ******")
avg_pred = get_mix_preds_by_result([bert_poss,xlnet_poss],"avg")
print(evaluate(real,avg_pred))
print("****** confidence prediction ******")
confidence_pred = get_mix_preds_by_result([bert_poss,xlnet_poss],"confidence")
print(evaluate(real,confidence_pred))
print("============================TWO MODEL:BERT+XLNET=================================")
#################BERT+XLNET MODEL#########################

#################BERT+FLAIR MODEL#########################
print("============================TWO MODEL:BERT+FLAIR=================================")
print("****** avg prediction ******")
avg_pred = get_mix_preds_by_result([bert_poss,flair_poss],"avg")
print(evaluate(real,avg_pred))
print("****** confidence prediction ******")
confidence_pred = get_mix_preds_by_result([bert_poss,flair_poss],"confidence")
print(evaluate(real,confidence_pred))
print("============================TWO MODEL:BERT+FLAIR=================================")
#################BERT+FLAIR MODEL#########################

#################ELMO+XLNET MODEL#########################
print("============================TWO MODEL:ELMO+XLNET=================================")
print("****** avg prediction ******")
avg_pred = get_mix_preds_by_result([elmo_poss,xlnet_poss],"avg")
print(evaluate(real,avg_pred))
print("****** confidence prediction ******")
confidence_pred = get_mix_preds_by_result([elmo_poss,xlnet_poss],"confidence")
print(evaluate(real,confidence_pred))
print("============================TWO MODEL:ELMO+XLNET=================================")
#################ELMO+XLNET MODEL#########################

#################ELMO+FLAIR MODEL#########################
print("============================TWO MODEL:ELMO+FLAIR=================================")
print("****** avg prediction ******")
avg_pred = get_mix_preds_by_result([elmo_poss,flair_poss],"avg")
print(evaluate(real,avg_pred))
print("****** confidence prediction ******")
confidence_pred = get_mix_preds_by_result([elmo_poss,flair_poss],"confidence")
print(evaluate(real,confidence_pred))
print("============================TWO MODEL:ELMO+FLAIR=================================")
#################ELMO+FLAIR MODEL#########################

#################XLNET+FLAIR MODEL#########################
print("============================TWO MODEL:XLNET+FLAIR=================================")
print("****** avg prediction ******")
avg_pred = get_mix_preds_by_result([xlnet_poss,flair_poss],"avg")
print(evaluate(real,avg_pred))
print("****** confidence prediction ******")
confidence_pred = get_mix_preds_by_result([xlnet_poss,flair_poss],"confidence")
print(evaluate(real,confidence_pred))
print("============================TWO MODEL:XLNET+FLAIR=================================")
#################XLNET+FLAIR MODEL#########################

#################BERT+ELMO+XLNET MODEL#########################
print("============================THREE MODEL:BERT+ELMO+XLNET=================================")
print("****** avg prediction ******")
avg_pred = get_mix_preds_by_result([bert_poss,elmo_poss,xlnet_poss],"avg")
print(evaluate(real,avg_pred))
print("****** confidence prediction ******")
confidence_pred = get_mix_preds_by_result([bert_poss,elmo_poss,xlnet_poss],"confidence")
print(evaluate(real,confidence_pred))
print("============================THREE MODEL:BERT+ELMO+XLNET=================================")
#################BERT+ELMO+XLNET MODEL#########################

#################BERT+ELMO+FLAIR MODEL#########################
print("============================THREE MODEL:BERT+ELMO+FLAIR=================================")
print("****** avg prediction ******")
avg_pred = get_mix_preds_by_result([bert_poss,elmo_poss,flair_poss],"avg")
print(evaluate(real,avg_pred))
print("****** confidence prediction ******")
confidence_pred = get_mix_preds_by_result([bert_poss,elmo_poss,flair_poss],"confidence")
print(evaluate(real,confidence_pred))
print("============================THREE MODEL:BERT+ELMO+FLAIR=================================")
#################BERT+ELMO+FLAIR MODEL#########################

#################BERT+XLNET+FLAIR MODEL#########################
print("============================THREE MODEL:BERT+XLNET+FLAIR=================================")
print("****** avg prediction ******")
avg_pred = get_mix_preds_by_result([bert_poss,flair_poss,xlnet_poss],"avg")
print(evaluate(real,avg_pred))
print("****** confidence prediction ******")
confidence_pred = get_mix_preds_by_result([bert_poss,flair_poss,xlnet_poss],"confidence")
print(evaluate(real,confidence_pred))
print("============================THREE MODEL:BERT+XLNET+FLAIR=================================")
#################BERT+XLNET+FLAIR MODEL#########################

#################ELMO+XLNET+FLAIR MODEL#########################
print("============================THREE MODEL:ELMO+XLNET+FLAIR=================================")
print("****** avg prediction ******")
avg_pred = get_mix_preds_by_result([flair_poss,elmo_poss,xlnet_poss],"avg")
print(evaluate(real,avg_pred))
print("****** confidence prediction ******")
confidence_pred = get_mix_preds_by_result([flair_poss,elmo_poss,xlnet_poss],"confidence")
print(evaluate(real,confidence_pred))
print("============================THREE MODEL:ELMO+XLNET+FLAIR=================================")
#################ELMO+XLNET+FLAIR MODEL#########################

#################BERT+ELMO+XLNET+FLAIR MODEL#########################
print("============================FOUR MODEL:BERT+ELMO+XLNET+FLAIR=================================")
print("****** avg prediction ******")
avg_pred = get_mix_preds_by_result([bert_poss,elmo_poss,xlnet_poss,flair_poss],"avg")
print(evaluate(real,avg_pred))
print("****** confidence prediction ******")
confidence_pred = get_mix_preds_by_result([bert_poss,elmo_poss,xlnet_poss,flair_poss],"confidence")
print(evaluate(real,confidence_pred))
print("============================FOUR MODEL:BERT+ELMO+XLNET+FLAIR=================================")
#################BERT+ELMO+XLNET+FLAIR MODEL#########################

#################8 MODEL#########################
print("============================8 MODEL:128+256=================================")
print("****** avg prediction ******")
avg_pred = get_mix_preds_by_result([bert_poss,bert128_poss,elmo_poss,elmo128_poss,xlnet_poss,xlnet128_poss,flair_poss,flair128_poss],"avg")
print(evaluate(real,avg_pred))
print("****** confidence prediction ******")
confidence_pred = get_mix_preds_by_result([bert_poss,bert128_poss,elmo_poss,elmo128_poss,xlnet_poss,xlnet128_poss,flair_poss,flair128_poss],"confidence")
print(evaluate(real,confidence_pred))
print("============================8 MODEL:128+256=================================")
#################8 MODEL#########################

#################16 MODEL#########################
print("============================16 MODEL:128+256+64+512=================================")
print("****** avg prediction ******")
avg_pred = get_mix_preds_by_result([bert_poss,bert128_poss,bert64_poss,bert512_poss,elmo_poss,elmo128_poss,elmo64_poss,elmo512_poss,xlnet_poss,xlnet128_poss,xlnet64_poss,xlnet512_poss,flair_poss,flair128_poss,flair64_poss,flair512_poss],"avg")
print(evaluate(real,avg_pred))
print("****** confidence prediction ******")
confidence_pred = get_mix_preds_by_result([bert_poss,bert128_poss,bert64_poss,bert512_poss,elmo_poss,elmo128_poss,elmo64_poss,elmo512_poss,xlnet_poss,xlnet128_poss,xlnet64_poss,xlnet512_poss,flair_poss,flair128_poss,flair64_poss,flair512_poss],"confidence")
print(evaluate(real,confidence_pred))
print("============================16 MODEL:128+256+64+512=================================")
#################16 MODEL#########################

#################BEST 8 MODEL#########################
print("============================1st BEST 8=================================")
print("****** avg prediction ******")
avg_pred = get_mix_preds_by_result([bert_poss,bert128_poss,bert64_poss,elmo_poss,elmo128_poss,elmo64_poss,elmo512_poss,flair_poss],"avg")
print(evaluate(real,avg_pred))
print("****** confidence prediction ******")
confidence_pred = get_mix_preds_by_result([bert_poss,bert128_poss,bert64_poss,elmo_poss,elmo128_poss,elmo64_poss,elmo512_poss,flair_poss],"confidence")
print(evaluate(real,confidence_pred))
print("============================1st BEST 8=================================")
#################BEST 8 MODEL#########################

#################BEST 8 MODEL#########################
print("============================2nd BEST 8=================================")
print("****** avg prediction ******")
avg_pred = get_mix_preds_by_result([bert_poss,bert128_poss,bert64_poss,elmo_poss,elmo128_poss,elmo64_poss,elmo512_poss,xlnet_poss],"avg")
print(evaluate(real,avg_pred))
print("****** confidence prediction ******")
confidence_pred = get_mix_preds_by_result([bert_poss,bert128_poss,bert64_poss,elmo_poss,elmo128_poss,elmo64_poss,elmo512_poss,xlnet_poss],"confidence")
print(evaluate(real,confidence_pred))
print("============================2nd BEST 8=================================")
#################BEST 8 MODEL#########################
