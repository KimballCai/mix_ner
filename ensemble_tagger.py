import torch
import numpy as np
from flair.models import SequenceTagger
from flair.data import Dictionary, Sentence, Token, Label, space_tokenizer
from flair.embeddings import HashEmbeddings
from typing import List


class EnsembleTagger(SequenceTagger):
	def __init__(self, models, tag_dictionary, tag_type, use_crf=True, mode='feature'):
		super().__init__(hidden_size=1,
						 embeddings=HashEmbeddings(),
						 tag_dictionary=tag_dictionary,
						 tag_type=tag_type,
						 use_crf=use_crf)
		self.__models = models
		self.__mode = mode


	def _calculate_loss(self, features: torch.tensor, sentences: List[Sentence]):
		if self.__mode == 'loss':
			losses = []
			for model in self.__models:
				losses.append(model._calculate_loss(features, sentences))
			return torch.mean(torch.stack(losses))
		else:
			return super()._calculate_loss(features, sentences)


	def forward(self, sentences: List[Sentence]):
		if self.__mode == 'feature':
			features = []
			for model in self.__models:
				features.append(model.forward(sentences))
			return torch.mean(torch.stack(features), dim=0)
		else:
			return super().forward(sentences)

	def __str__(self):
		return 'Ensemble Tagger: [\n' + ',\n'.join([str(model) for model in self.__models]) + '\n]'
