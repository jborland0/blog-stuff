import json
import numpy as np
from collections import Counter
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

# open file of sentences
f = open("C:\\Users\\John\\Documents\\python\\pytorch\\facebook_sentences.json", "r")

# load sentences as json
sentences = json.load(f)

# take 1100 sentences
sentences = sentences[:1100]

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
# print(str(len(sorted_words)) + " total words.")

# create word-to-int mapping
vocab_to_int={w:i+1 for i,(w,c) in enumerate(sorted_words)}
# print(vocab_to_int)

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
labels=[]

# copy each sentence into the label vector
for sentence in sentences:
	if sentence["grateful"] == 1:
		labels.append(0)
	elif sentence["happy"] == 1:
		labels.append(1)
	elif sentence["hopeful"] == 1:
		labels.append(2)
	elif sentence["determined"] == 1:
		labels.append(3)
	elif sentence["aware"] == 1:
		labels.append(4)
	elif sentence["stable"] == 1:
		labels.append(5)
	elif sentence["frustrated"] == 1:
		labels.append(6)
	elif sentence["overwhelmed"] == 1:
		labels.append(7)
	elif sentence["angry"] == 1:
		labels.append(8)
	elif sentence["guilty"] == 1:
		labels.append(9)
	elif sentence["lonely"] == 1:
		labels.append(10)
	elif sentence["scared"] == 1:
		labels.append(11)
	elif sentence["sad"] == 1:
		labels.append(12)

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
batch_size=55
train_loader=DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader=DataLoader(valid_data, batch_size=batch_size, shuffle=True)
test_loader=DataLoader(test_data, batch_size=batch_size, shuffle=True)

# define LSTM class
class SentimentalLSTM(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):    
        """
        Initialize the model by setting up the layers
        """
        super().__init__()
        self.output_size=output_size
        self.n_layers=n_layers
        self.hidden_dim=hidden_dim
        
        #Embedding and LSTM layers
        self.embedding=nn.Embedding(vocab_size, embedding_dim)
        self.lstm=nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        
        #dropout layer
        self.dropout=nn.Dropout(0.3)
        
        #Linear and sigmoid layer
        self.fc1=nn.Linear(hidden_dim, 64)
        self.fc2=nn.Linear(64, 16)
        self.fc3=nn.Linear(16,output_size)
        self.sigmoid=nn.Sigmoid()
        
    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size=x.size(0)
        
        #Embadding and LSTM output
        embedd=self.embedding(x)
        lstm_out, hidden=self.lstm(embedd, hidden)
        
        #stack up the lstm output
        lstm_out=lstm_out.contiguous().view(-1, self.hidden_dim)
        
        #dropout and fully connected layers
        out=self.dropout(lstm_out)
        out=self.fc1(out)
        out=self.dropout(out)
        out=self.fc2(out)
        out=self.dropout(out)
        out=self.fc3(out)
        sig_out=self.sigmoid(out)
		
        sig_out=sig_out.view(batch_size, -1)
        sig_out=sig_out[:, -1]
        
        return sig_out, hidden
    
    def init_hidden(self, batch_size):
        """Initialize Hidden STATE"""
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden

# Instantiate the model w/ hyperparams
vocab_size = len(vocab_to_int)+1 # +1 for the 0 padding
output_size = 13
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
