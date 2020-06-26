import json
import numpy as np
from collections import Counter
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from nn.elmolstm import ELMoLSTM
import pickle
import pandas as pd

def train_mood(mood):
	# read data
	train = pd.read_csv('elmo/train_' + mood + '.csv')
	labels = train['label'].values

	# load elmo_train_new
	pickle_in = open('elmo/elmo_train_' + mood + '.pickle', 'rb')
	features = pickle.load(pickle_in)

	# split dataset into training, test, validation
	train_x=features[:int(0.8*len(features))]
	train_y=labels[:int(0.8*len(features))]
	valid_x=features[int(0.8*len(features)):int(0.9*len(features))]
	valid_y=labels[int(0.8*len(features)):int(0.9*len(features))]
	test_x=features[int(0.9*len(features)):]
	test_y=labels[int(0.9*len(features)):]

	#create Tensor Dataset
	train_data=TensorDataset(torch.FloatTensor(train_x), torch.FloatTensor(train_y))
	valid_data=TensorDataset(torch.FloatTensor(valid_x), torch.FloatTensor(valid_y))
	test_data=TensorDataset(torch.FloatTensor(test_x), torch.FloatTensor(test_y))

	#dataloader
	batch_size=1
	train_loader=DataLoader(train_data, batch_size=batch_size, shuffle=True)
	valid_loader=DataLoader(valid_data, batch_size=batch_size, shuffle=True)
	test_loader=DataLoader(test_data, batch_size=batch_size, shuffle=True)

	# Instantiate the model w/ hyperparams
	output_size = 1
	embedding_dim = 1024
	hidden_dim = 256
	n_layers = 2

	net = ELMoLSTM(output_size, embedding_dim, hidden_dim, n_layers)
	print(net)

	# loss and optimization functions
	lr=0.001

	criterion = nn.BCELoss()
	optimizer = torch.optim.Adam(net.parameters(), lr=lr)

	# check if CUDA is available
	train_on_gpu = torch.cuda.is_available()

	# training params

	epochs = 3 # 3-4 is approx where I noticed the validation loss stop decreasing

	counter = 0
	print_every = 100
	clip=5 # gradient clipping

	# move model to GPU, if available
	if(train_on_gpu):
		net.cuda()

	net.train()
	# train for some number of epochs
	for e in range(epochs):
		# initialize hidden state
		h = net.init_hidden(batch_size)

		# batch loop
		for inputs, labels in train_loader:
			counter += 1

			if(train_on_gpu):
				inputs=inputs.cuda()
				labels=labels.cuda()
			# Creating new variables for the hidden state, otherwise
			# we'd backprop through the entire training history
			h = tuple([each.data for each in h])

			# zero accumulated gradients
			net.zero_grad()

			# get the output from the model
			output, h = net(inputs, h)

			# calculate the loss and perform backprop
			loss = criterion(output.squeeze(), labels.float())
			loss.backward()
			# `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
			nn.utils.clip_grad_norm_(net.parameters(), clip)
			optimizer.step()

			# loss stats
			if counter % print_every == 0:
				# Get validation loss
				val_h = net.init_hidden(batch_size)
				val_losses = []
				net.eval()
				for inputs, labels in valid_loader:

					# Creating new variables for the hidden state, otherwise
					# we'd backprop through the entire training history
					val_h = tuple([each.data for each in val_h])

					if(train_on_gpu):
						inputs, labels = inputs.cuda(), labels.cuda()

					output, val_h = net(inputs, val_h)
					val_loss = criterion(output.squeeze(), labels.float())

					val_losses.append(val_loss.item())

				net.train()
				print("Epoch: {}/{}...".format(e+1, epochs),
					  "Step: {}...".format(counter),
					  "Loss: {:.6f}...".format(loss.item()),
					  "Val Loss: {:.6f}".format(np.mean(val_losses)))

	test_losses = [] # track loss
	num_correct = 0

	# init hidden state
	h = net.init_hidden(batch_size)

	net.eval()
	# iterate over test data
	for inputs, labels in test_loader:

		# Creating new variables for the hidden state, otherwise
		# we'd backprop through the entire training history
		h = tuple([each.data for each in h])

		if(train_on_gpu):
			inputs, labels = inputs.cuda(), labels.cuda()


		output, h = net(inputs, h)

		# calculate loss
		test_loss = criterion(output.squeeze(), labels.float())
		test_losses.append(test_loss.item())

		# convert output probabilities to predicted class (0 or 1)
		pred = torch.round(output.squeeze())  # rounds to the nearest integer

		# compare predictions to true label
		correct_tensor = pred.eq(labels.float().view_as(pred))
		correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
		num_correct += np.sum(correct)


	# -- stats! -- ##
	# avg test loss
	print("Test loss: {:.3f}".format(np.mean(test_losses)))

	# accuracy over all test data
	test_acc = num_correct/len(test_loader.dataset)
	print("Test accuracy: {:.3f}".format(test_acc))

	# save state dictionary to file
	torch.save(net.state_dict(), mood + '.pt')

train_mood('grateful')
train_mood('happy')
train_mood('hopeful')
train_mood('determined')
train_mood('aware')
train_mood('stable')
train_mood('frustrated')
train_mood('overwhelmed')
train_mood('angry')
train_mood('guilty')
train_mood('lonely')
train_mood('scared')
train_mood('sad')
