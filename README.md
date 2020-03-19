# mix_ner 

using flair tools (https://github.com/flairNLP/flair.git)

## ELMo, BERT, Flair, and XLNet.

# Instructions
## Requirement
pytorch
flair
allennlp

## Usage
### Train tagger with (ELMo/BERT/Flair) Embedding
```
python train.py --embed=elmo
```

### Get vote prediction
Several models co-predict the final result by voting the label-position(B/I) and label-classification(LOC/PER/ORG/MISC) separately.
```
python vote_predict.py
```


## Problem
### Elmo
1. UnboundLocalError: local variable 'allennlp' referenced before assignment  
**solution:** pip install allennlp

### BERT
1. ImportError: FloatProgress not found. Please update jupyter and ipywidgets  
**solution:** conda install -c conda-forge ipywidgets

