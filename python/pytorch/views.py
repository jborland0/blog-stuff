from django.shortcuts import render
from django.http import HttpResponse
import json
import numpy as np
from collections import Counter
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from nn.sentimentalltsm import SentimentalLSTM
import pickle

def index(request):
	context = {}
	return render(request, 'nn/index.html')

def results(request):
	text = request.POST.get('text', '')
	response = "You said: '" + text + "'<br /><br />"
	response += "Here's how I think you are feeling: <br /><br />"
	response += "Grateful " + str(moodilyze(text, 'grateful')) + "<br />"
	response += "Happy " + str(moodilyze(text, 'happy')) + "<br />"
	response += "Hopeful " + str(moodilyze(text, 'hopeful')) + "<br />"
	response += "Determined " + str(moodilyze(text, 'determined')) + "<br />"
	response += "Aware " + str(moodilyze(text, 'aware')) + "<br />"
	response += "Stable " + str(moodilyze(text, 'stable')) + "<br />"
	response += "Frustrated " + str(moodilyze(text, 'frustrated')) + "<br />"
	response += "Overwhelmed " + str(moodilyze(text, 'overwhelmed')) + "<br />"
	response += "Guilty " + str(moodilyze(text, 'guilty')) + "<br />"
	response += "Angry " + str(moodilyze(text, 'angry')) + "<br />"
	response += "Lonely " + str(moodilyze(text, 'lonely')) + "<br />"
	response += "Scared " + str(moodilyze(text, 'scared')) + "<br />"
	response += "Sad " + str(moodilyze(text, 'sad')) + "<br />"
	response += "<br /><a href='/moodilyzer/nn/'>Try Again!</a>"
	return HttpResponse(response)

def moodilyze(text,emotion):
	# Instantiate the model w/ hyperparams
	vocab_size = 2576
	output_size = 1
	embedding_dim = 400
	hidden_dim = 256
	n_layers = 2
	batch_size = 1

	net = SentimentalLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

	# read the state dictionary from the file
	net.load_state_dict(torch.load('/var/www/jborland/html/moodilyzer/moodilyzer/nn/models/' + emotion + '.pt'))

	# init hidden state
	h = net.init_hidden(batch_size)

	# set dropout and batch normalization layers to evaluation mode
	net.eval()

	# read vocabulary from file
	vocab_to_int = pickle.load( open( "/var/www/jborland/html/moodilyzer/moodilyzer/nn/models/vocab_to_int.p", "rb" ) )

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
