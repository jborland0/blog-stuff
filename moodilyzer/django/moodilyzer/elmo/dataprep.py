import json
import numpy as np
from collections import Counter
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import pickle

# open file of sentences
f = open("C:\\Users\\John\\Documents\\python\\pytorch\\facebook_sentences.json", "r")

# load sentences as json
sentences = json.load(f)
print(str(len(sentences)) + " sentences.")

# function to write a training set for a mood
def write_training_set(mood):
	# initialize "tweet" index
	tweetIdx = 1
	
	# write training file
	with open('train_' + mood + '.csv', 'w') as f:
		f.write('id,label,tweet\n')
		for sentence in sentences:
			f.write(str(tweetIdx) + ',' + str(sentence[mood]) + ',"' + sentence["sentence"] + '"\n')
			tweetIdx += 1
	print("wrote " + str(tweetIdx - 1) + " training sentences.")		

write_training_set('grateful')
write_training_set('happy')
write_training_set('hopeful')
write_training_set('determined')
write_training_set('aware')
write_training_set('stable')
write_training_set('frustrated')
write_training_set('overwhelmed')
write_training_set('angry')
write_training_set('guilty')
write_training_set('lonely')
write_training_set('scared')
write_training_set('sad')
