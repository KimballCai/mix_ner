from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import ELMoEmbeddings,BertEmbeddings,FlairEmbeddings,XLNetEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from bagging_tagger import EnsembleTagger
from flair.trainers import ModelTrainer
from eval.conlleval import evaluate
import argparse

def parse_args():
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument('--model', type=str)
	arg_parser.add_argument('--batch_size', default=32, type=int)
	arg_parser.add_argument('--epoch', default=50, type=int)
	arg_parser.add_argument('--restore', action="store_true")
	arg_parser.add_argument('--train', action="store_true")
	arg_parser.add_argument('--lr', default=0.01, type=float);
	# arg_parser.add_argument('--gpu', default='0')
	return arg_parser.parse_args()

ARGS = parse_args()

model_path = "/hdd1/kurisu/cs6207/log/ensemble/" + ARGS.model +"/"

# os.environ["CUDA_VISIBLE_DEVICES"] = ARGS.gpu

columns = {0: 'text', 1: '_', 2: '_', 3: 'ner'}

data_folder = './data'

corpus: Corpus = ColumnCorpus(data_folder, columns,
							  train_file='train.txt',
							  test_file='test.txt',
							  dev_file='valid.txt')
print(corpus)
tag_type = 'ner'
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary)

if ARGS.restore:
	ensemble_tagger = EnsembleTagger.load(model_path + "final-model.pt")
else:
	elmo_tagger = SequenceTagger(hidden_size=256,
								 embeddings=ELMoEmbeddings('small'),
								 tag_dictionary=tag_dictionary,
								 tag_type=tag_type,
								 use_crf=True)
	bert_tagger = SequenceTagger(hidden_size=256,
								 embeddings=BertEmbeddings(),
								 tag_dictionary=tag_dictionary,
								 tag_type=tag_type,
								 use_crf=True)
	xlnet_tagger = SequenceTagger(hidden_size=256,
								  embeddings=XLNetEmbeddings(),
								  tag_dictionary=tag_dictionary,
								  tag_type=tag_type,
								  use_crf=True)
	flair_tagger = SequenceTagger(hidden_size=256,
								  embeddings=StackedEmbeddings([FlairEmbeddings('news-forward'),FlairEmbeddings('news-backward')]),
								  tag_dictionary=tag_dictionary,
								  tag_type=tag_type,
								  use_crf=True)
	models = []
	if ARGS.model == "be":
		models = [bert_tagger, elmo_tagger]
	elif ARGS.model == "bf":
		models = [bert_tagger, flair_tagger]
	elif ARGS.model == "bx":
		models = [bert_tagger, xlnet_tagger]
	elif ARGS.model == "bef":
		models = [bert_tagger, elmo_tagger, flair_tagger]
	elif ARGS.model == "bex":
		models = [bert_tagger, elmo_tagger, xlnet_tagger]
	elif ARGS.model == "bfx":
		models = [bert_tagger, flair_tagger, xlnet_tagger]
	elif ARGS.model == "befx":
		models = [bert_tagger, elmo_tagger, flair_tagger, xlnet_tagger]

	ensemble_tagger = EnsembleTagger(models=models,
									 tag_type=tag_type,
									 mode='loss')
if ARGS.train:
	trainer: ModelTrainer = ModelTrainer(ensemble_tagger, corpus)

	trainer.train(model_path,
				  learning_rate=ARGS.lr,
				  mini_batch_size=ARGS.batch_size,
				  max_epochs=ARGS.epoch)

real = []
for sentence in corpus.test:
	for token in sentence.tokens:
		real.append(token.get_tag("ner").value)

def test(model, data):
	results = []
	for sentence in data:
		model.predict(sentence,all_tag_prob=True)
		for token in sentence.tokens:
			results.append(token.get_tag("ner").value)
	return results

ensemble_pred = test(ensemble_tagger, corpus.test)
print(evaluate(real, ensemble_pred))

