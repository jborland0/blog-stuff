import os
import numpy as np
from collections import Counter
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from lstm.sentimentalltsm import SentimentalLSTM
import pickle

class LSTMClassifier:
	def __init__(self, datadir):
		self.datadir = datadir

	def moodilyze(self, text, emotion):
		# Instantiate the model w/ hyperparams
		vocab_size = 2576
		output_size = 1
		embedding_dim = 400
		hidden_dim = 256
		n_layers = 2
		batch_size = 1

		net = SentimentalLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

		# read the state dictionary from the file
		net.load_state_dict(torch.load(os.path.join(self.datadir, emotion + '.pt')))

		# init hidden state
		h = net.init_hidden(batch_size)

		# set dropout and batch normalization layers to evaluation mode
		net.eval()

		# read vocabulary from file
		vocab_to_int = pickle.load( open(os.path.join(self.datadir, "vocab_to_int.p"), "rb" ) )

		sentences = list()
		sentences.append(text)
		encoded_reviews=list()
		for sentence in sentences:
			encoded_review=list()
			for word in sentence.split():
				if word not in vocab_to_int.keys():
					#if word is not available in self.vocab_to_int put 0 in that place
					encoded_review.append(0)
				else:
					encoded_review.append(vocab_to_int[word])
			encoded_reviews.append(encoded_review)

		# pad with zeros
		sequence_length=44
		features=np.zeros((len(encoded_reviews), sequence_length), dtype=int)
		for i, review in enumerate(encoded_reviews):
			review_len=len(review)
			if (review_len<=sequence_length):
				zeros=list(np.zeros(sequence_length-review_len))
				new=zeros+review
			else:
				new=review[:sequence_length]
			features[i,:]=np.array(new)

		# Creating new variables for the hidden state, otherwise
		# we'd backprop through the entire training history
		h = tuple([each.data for each in h])

		if(torch.cuda.is_available()):
			inputs, labels = inputs.cuda(), labels.cuda()

		output, h = net(torch.tensor(features).type(torch.LongTensor), h)

		# get a readable form of the output
		readout = output.data.cpu().numpy()

		if readout[0] < 0.01:
			return 0
		if readout[0] > 1:
			return 100
		return readout[0] * 100
