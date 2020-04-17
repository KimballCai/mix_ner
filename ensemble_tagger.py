import torch
import numpy as np
import flair
from flair.models import SequenceTagger
from flair.data import Dictionary, Sentence, Token, Label, space_tokenizer
from flair.training_utils import Metric, Result, store_embeddings
from typing import List, Union, Optional, Callable, Dict
from copy import copy, deepcopy
# from vote_predict import vote


def pad_tensors(tensor_list):
	ml = max([x.shape[0] for x in tensor_list])
	shape = [len(tensor_list), ml] + list(tensor_list[0].shape[1:])
	template = torch.zeros(*shape, dtype=torch.long, device=flair.device)
	lens_ = [x.shape[0] for x in tensor_list]
	for i, tensor in enumerate(tensor_list):
		template[i, : lens_[i]] = tensor

	return template, lens_


def _forward(model, sentences: List[Sentence]):

	model.embeddings.embed(sentences)

	# print([len(sentence) for sentence in sentences])
	# print('embedding_length: %d' % model.embeddings.embedding_length)

	lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
	longest_token_sequence_in_batch: int = max(lengths)

	pre_allocated_zero_tensor = torch.zeros(
		model.embeddings.embedding_length * longest_token_sequence_in_batch,
		dtype=torch.float,
		device=flair.device,
	)

	all_embs = list()
	for sentence in sentences:
		all_embs += [
			emb for token in sentence for emb in token.get_each_embedding()
		]
		nb_padding_tokens = longest_token_sequence_in_batch - len(sentence)

		if nb_padding_tokens > 0:
			t = pre_allocated_zero_tensor[
				: model.embeddings.embedding_length * nb_padding_tokens
			]
			all_embs.append(t)
			# print('length of t: %d' % len(t))
		# print('nb_padding_tokens: %d, all_embs length: %d' % (nb_padding_tokens, np.sum([len(emb) for emb in all_embs])))
	sentence_tensor = torch.cat(all_embs).view(
		[
			len(sentences),
			longest_token_sequence_in_batch,
			model.embeddings.embedding_length,
		]
	)

	# --------------------------------------------------------------------
	# FF PART
	# --------------------------------------------------------------------
	if model.use_dropout > 0.0:
		sentence_tensor = model.dropout(sentence_tensor)
	if model.use_word_dropout > 0.0:
		sentence_tensor = model.word_dropout(sentence_tensor)
	if model.use_locked_dropout > 0.0:
		sentence_tensor = model.locked_dropout(sentence_tensor)

	if model.relearn_embeddings:
		sentence_tensor = model.embedding2nn(sentence_tensor)

	if model.use_rnn:
		packed = torch.nn.utils.rnn.pack_padded_sequence(
			sentence_tensor, lengths, enforce_sorted=False, batch_first=True
		)

		# if initial hidden state is trainable, use this state
		if model.train_initial_hidden_state:
			initial_hidden_state = [
				model.lstm_init_h.unsqueeze(1).repeat(1, len(sentences), 1),
				model.lstm_init_c.unsqueeze(1).repeat(1, len(sentences), 1),
			]
			rnn_output, hidden = model.rnn(packed, initial_hidden_state)
		else:
			rnn_output, hidden = model.rnn(packed)

		sentence_tensor, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
			rnn_output, batch_first=True
		)

		if model.use_dropout > 0.0:
			sentence_tensor = model.dropout(sentence_tensor)
		# word dropout only before LSTM - TODO: more experimentation needed
		# if self.use_word_dropout > 0.0:
		#     sentence_tensor = self.word_dropout(sentence_tensor)
		if model.use_locked_dropout > 0.0:
			sentence_tensor = model.locked_dropout(sentence_tensor)

	features = model.linear(sentence_tensor)

	return features
	

def _calculate_loss(model, features: torch.tensor, sentences: List[Sentence]):
	
	lengths: List[int] = [len(sentence.tokens) for sentence in sentences]

	tag_list: List = []
	for s_id, sentence in enumerate(sentences):
		# get the tags in this sentence
		tag_idx: List[int] = [
			model.tag_dictionary.get_idx_for_item(token.get_tag(model.tag_type).value)
			for token in sentence
		]
		# add tags as tensor
		tag = torch.tensor(tag_idx, device=flair.device)
		tag_list.append(tag)

	if model.use_crf:
		# pad tags if using batch-CRF decoder
		tags, _ = pad_tensors(tag_list)

		forward_score = model._forward_alg(features, lengths)
		gold_score = model._score_sentence(features, tags, lengths)

		score = forward_score - gold_score

		return score.mean()

	else:
		score = 0
		for sentence_feats, sentence_tags, sentence_length in zip(
			features, tag_list, lengths
		):
			sentence_feats = sentence_feats[:sentence_length]
			score += torch.nn.functional.cross_entropy(
				sentence_feats, sentence_tags, weight=model.loss_weights
			)
		score /= len(features)
		return score


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
	# classificatin_vote = {'ORG': 0, 'PER': 0, 'MISC': 0, 'LOC': 0, 'O': 0}
	# position_vote = {'B': 0, 'I': 0, 'O': 0}
	# for vote_tag in vote_list:
	# 	tag = 'O' if vote_tag == '<unk>' or vote_tag == '<START>' or vote_tag == '<STOP>' else vote_tag
	# 	temp_position = get_position_tag(tag)
	# 	position_vote[temp_position] = position_vote[temp_position] + 1

	# 	temp_classification = get_classification_tag(tag)
	# 	classificatin_vote[temp_classification] = classificatin_vote[temp_classification] + 1

	# final_position = max(position_vote, key=position_vote.get)
	# final_classificatin = max(classificatin_vote, key=classificatin_vote.get)
	# if (final_position == 'O') and (final_classificatin == 'O'):
	# 	return 'O'
	# return final_position + "-" + final_classificatin
	classification_vote = {'B-ORG': 0, 'B-PER': 0, 'B-MISC': 0, 'B-LOC': 0, 'I-ORG': 0, 'I-PER': 0, 'I-MISC': 0, 'I-LOC': 0, 'O': 0}
	for vote_tag in vote_list:
		classification_vote[vote_tag] += 1
		final_classification = max(classification_vote, key=classification_vote.get)
		return final_classification


def _predict_batch(models, batch):
	with torch.no_grad():
		losses = []
		tags = []
		for i, model in enumerate(models):
			sentences = deepcopy(batch)
			# features = model.forward(sentences)
			# losses.append(model._calculate_loss(features, sentences))
			# batch_tag, _ = model._obtain_labels(
			# 	feature=features,
			# 	batch_sentences=sentences,
			# 	transitions=model.transitions,
			# 	get_all_tags=False,
			# )
			losses.append(model.forward_loss(sentences).detach().cpu().numpy())
			model.predict(sentences)
			batch_tag = [[token.get_tag("ner").value for token in sentence] for sentence in sentences]
			tags.append(batch_tag)
		tags = [[Label(vote(token)) for token in sentence]
				for sentence in [list(zip(*x)) for x in list(zip(*tags))]]

	return np.mean(losses), tags


def _predict_sentence(models, sentence):
	with torch.no_grad():
		tags = [] # [List[List[Label]]]
		for i, model in enumerate(models):
			seq = deepcopy(sentence)
			model.predict(seq)
			batch_tag = [token.get_tag("ner").value for token in seq]
			tags.append(batch_tag)
		tags = [Label(vote(token)) for token in np.transpose(tags, (1,0))]

	return tags


class EnsembleTagger(flair.nn.Model):
	def __init__(self, models, tag_type, mode='loss'):
		super().__init__()
		self.__models = torch.nn.ModuleList(models)
		self.__mode = mode
		self.tag_type = tag_type


	def forward_loss(self, data_points):
		if self.__mode == 'feature':
			features = []
			for model in self.__models:
				sentences = deepcopy(data_points)
				features.append(_forward(model, sentences))
			features = torch.mean(torch.stack(features), dim=0)
			return _calculate_loss(self.__models[0], features, data_points)
		elif self.__mode == 'loss':
			losses = []
			for i, model in enumerate(self.__models):
				sentences = deepcopy(data_points)
				features = _forward(model, sentences)
				losses.append(_calculate_loss(model, features, sentences))
			return torch.mean(torch.stack(losses))


	def evaluate(self, data_loader, out_path=None, embedding_storage_mode="none"):
		eval_loss = 0

		batch_no: int = 0

		metric = Metric("Evaluation", beta=1.0)

		# lines: List[str] = []

		for batch in data_loader:
			batch_no += 1

			loss, tags = _predict_batch(self.__models, batch)

			eval_loss += loss

			for (sentence, sent_tags) in zip(batch, tags):
				for (token, tag) in zip(sentence.tokens, sent_tags):
					token: Token = token
					token.add_tag("predicted", tag.value, tag.score)

					# append both to file for evaluation
				#     eval_line = "{} {} {} {}\n".format(
				#         token.text,
				#         token.get_tag(self.tag_type).value,
				#         tag.value,
				#         tag.score,
				#     )
				#     lines.append(eval_line)
				# lines.append("\n")

			for sentence in batch:
				# make list of gold tags
				gold_tags = [
					(tag.tag, tag.text) for tag in sentence.get_spans(self.tag_type)
				]
				# make list of predicted tags
				predicted_tags = [
					(tag.tag, tag.text) for tag in sentence.get_spans("predicted")
				]

				# check for true positives, false positives and false negatives
				for tag, prediction in predicted_tags:
					if (tag, prediction) in gold_tags:
						metric.add_tp(tag)
					else:
						metric.add_fp(tag)

				for tag, gold in gold_tags:
					if (tag, gold) not in predicted_tags:
						metric.add_fn(tag)
					else:
						metric.add_tn(tag)

			store_embeddings(batch, embedding_storage_mode)

		eval_loss /= batch_no

		detailed_result = (
			f"\nMICRO_AVG: acc {metric.micro_avg_accuracy()} - f1-score {metric.micro_avg_f_score()}"
			f"\nMACRO_AVG: acc {metric.macro_avg_accuracy()} - f1-score {metric.macro_avg_f_score()}"
		)
		for class_name in metric.get_classes():
			detailed_result += (
				f"\n{class_name:<10} tp: {metric.get_tp(class_name)} - fp: {metric.get_fp(class_name)} - "
				f"fn: {metric.get_fn(class_name)} - tn: {metric.get_tn(class_name)} - precision: "
				f"{metric.precision(class_name):.4f} - recall: {metric.recall(class_name):.4f} - "
				f"accuracy: {metric.accuracy(class_name):.4f} - f1-score: "
				f"{metric.f_score(class_name):.4f}"
			)

		result = Result(
			main_score=metric.micro_avg_f_score(),
			log_line=f"{metric.precision()}\t{metric.recall()}\t{metric.micro_avg_f_score()}",
			log_header="PRECISION\tRECALL\tF1",
			detailed_results=detailed_result,
		)

		return result, eval_loss


	def predict(self, sentence, all_tag_prob=False, embedding_storage_mode="none"):
		
		with torch.no_grad():

			result = sentence

			tags = _predict_sentence(self.__models, sentence)

			for (token, tag) in zip(sentence.tokens, tags):
				token.add_tag_label(self.tag_type, tag)

			# clearing token embeddings to save memory
			store_embeddings(sentence, storage_mode=embedding_storage_mode)

			result = sentence
			assert len(sentence) == len(result)
			return result


	def _get_state_dict(self):
		models_dict = []
		for model in self.__models:
			models_dict.append(model._get_state_dict())
		model_state = {
			"state_dict": self.state_dict(),
			"models": models_dict,
			"mode": self.__mode,
			"tag_type": self.tag_type
		}
		return model_state


	@staticmethod
	def _init_model_with_state_dict(state):
		mode = 'loss' if "mode" not in state.keys() else state["mode"]
		tag_type = 'ner' if "tag_type" not in state.keys() else state["tag_type"]
		models = [SequenceTagger._init_model_with_state_dict(model_state) for model_state in state["models"]]
		model = EnsembleTagger(models=models,
							   tag_type=tag_type,
							   mode=mode)
		model.load_state_dict(state["state_dict"])
		return model


	def __str__(self):
		return 'Ensemble Tagger: [\n' + ',\n'.join([str(model) for model in self.__models]) + '\n]'
