import json
import numpy as np
from collections import Counter
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from nn.sentimentalltsm import SentimentalLSTM
import pickle

# open file of sentences
f = open("C:\\Users\\John\\Documents\\python\\pytorch\\facebook_sentences.json", "r")

# load sentences as json
sentences = json.load(f)

# take 1100 sentences
#sentences = sentences[:1100]

print(str(len(sentences)) + " sentences.")

# create dictionary for word counts
wordcounts = {}

# join all sentences together
all_text = ""
for sentence in sentences:
	all_text += " " + sentence["sentence"]
	
# get word counts
all_words = all_text.split()
count_words = Counter(all_words)
total_words=len(all_words)
sorted_words=count_words.most_common(total_words)
# print("Top ten occuring words:")
# print(sorted_words[:10])
print(str(len(sorted_words)) + " total words.")

# create word-to-int mapping
vocab_to_int={w:i+1 for i,(w,c) in enumerate(sorted_words)}

# save word-to-int mapping to file
pickle.dump( vocab_to_int, open( "vocab_to_int.p", "wb" ) )

# longest_len = 0
# get length of longest sentence (44)
# for sentence in sentences:
#	words = sentence["sentence"].split()
#	if len(words) > longest_len:
#		longest_len = len(words)
# print("longest sentence is " + str(longest_len) + " words")

# encode sentences into integers
encoded_reviews=list()
for sentence in sentences:
	encoded_review=list()
	for word in sentence["sentence"].split():
		if word not in vocab_to_int.keys():
			#if word is not available in vocab_to_int put 0 in that place
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

# create a label vector
labels=[sentence["sad"] for sentence in sentences]

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
vocab_size = len(vocab_to_int)+1 # +1 for the 0 padding
output_size = 1
embedding_dim = 400
hidden_dim = 256
n_layers = 2

net = SentimentalLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
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
        output, h = net(inputs.type(torch.LongTensor), h)

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

                output, val_h = net(inputs.type(torch.LongTensor), val_h)
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


	output, h = net(inputs.type(torch.LongTensor), h)

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
torch.save(net.state_dict(), 'sad.pt')
