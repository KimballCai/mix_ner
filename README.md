# Combine Contextual Embedding in NER
NUS CS6207 Course Project   
Developed by Qingpeng Cai, Hengchang Hu, Zhengda Bian  
May 2020

## Requirement
> [flair](https://github.com/flairNLP/flair)  
> pytorch   
> allennlp  

for more information, please refer to [requirement.txt](./requirement.txt)

## Quick Start
### train tagger with (ELMo/BERT/Flair/XLNet) Embedding
```bash
python train.py --embed bert 
```
### test
```bash
python test.py --path "./log/elmo_20200330002549_256/"
```
### Prediction Ensembling
- Avg-agg & Con-agg
Get prediction probability output of single model. The result (porbability) will be saved automatically into the folder /`result/prob/`
```bash
python model_avg_conf_result.py
```

Then we can get the **average prbability ensemble prediction** and **highest confidence ensemble prediction** by excute the command line below. The F1 result will be saved into log file.
```bash
python -u avg_conf_predict.py > result_avg_confidence 2>&1 &
```

- N-Vote & H-Vote
Get result of single model. The result (tag list) will be saved automatically into the folder `/result/tag/`
```bash
python model_vote_result.py
```

Then we can get the **naive vote ensemble prediction** and **hierarchy ensemble prediction** by excute the command line below. The F1 result will be saved into log file.
```bash
python -u vote_predict.py > result_vote.log 2>&1 &
```

- Con-Vote
We can reuse the ouput result from `/result/tag/`. We can get the **voting with confidence threshold ensemble prediction** by excute the command line below. The F1 result will be saved into log file.
```bash
python -u conf_vote_predict.py > result_conf_vote.log 2>&1 &
```

### Bagging
```bash
python train_ensemble.py --train --model=befx 
```
add `--restore` if you want to continue train on the pretrained model.  
delete `--train` if you just want test the model.

### Embedding Stacking
```bash
python train.py --embed mix_bexf --hiddensize 256
```
and then use the `eval/model_result.py` to evaluate the results.

## Contact
If you need the pretrained model or have other questions, please email your questions or comments to 
qingpeng@comp.nus.edu.sg, holdenhhc@gmail.com, and bian.zhengda@u.nus.edu


## Problem
### Elmo
1. UnboundLocalError: local variable 'allennlp' referenced before assignment  
**solution:** pip install allennlp

### BERT
1. ImportError: FloatProgress not found. Please update jupyter and ipywidgets  
**solution:** conda install -c conda-forge ipywidgets

