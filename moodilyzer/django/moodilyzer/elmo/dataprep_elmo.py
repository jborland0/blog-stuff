import tensorflow_hub as hub
import tensorflow.compat.v1 as tf
import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm
import re
import time
import pickle
pd.set_option('display.max_colwidth', 200)

#To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
tf.disable_eager_execution()

# create elmo
print("loading elmo...")
elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
print("elmo loaded.")

# import spaCy's language model
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# function to lemmatize text
def lemmatization(texts):
	output = []
	for i in texts:
		s = [token.lemma_ for token in nlp(i)]
		output.append(' '.join(s))
	return output

# function to return average elmo vector for each sentence in x
def elmo_vectors(x):
	embeddings = elmo(x.tolist(), signature="default", as_dict=True)["elmo"]

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		sess.run(tf.tables_initializer())
		# return average of ELMo features
		return sess.run(tf.reduce_mean(embeddings,1))

def write_elmo_vectors(mood):
	# read data
	train = pd.read_csv('train_' + mood + '.csv')

	# remove URL's from train
	train['clean_tweet'] = train['tweet'].apply(lambda x: re.sub(r'http\S+', '', x))

	# remove punctuation marks
	punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'

	train['clean_tweet'] = train['clean_tweet'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))

	# convert text to lowercase
	train['clean_tweet'] = train['clean_tweet'].str.lower()

	# remove numbers
	train['clean_tweet'] = train['clean_tweet'].str.replace("[0-9]", " ")

	# remove whitespaces
	train['clean_tweet'] = train['clean_tweet'].apply(lambda x:' '.join(x.split()))

	# lemmatize text
	train['clean_tweet'] = lemmatization(train['clean_tweet'])

	# chop up sentences into batches of 100
	list_train = [train[i:i+100] for i in range(0,train.shape[0],100)]

	# Extract ELMo embeddings
	print("creating elmo training vectors...")
	elmo_train = [elmo_vectors(x['clean_tweet']) for x in list_train]

	# concatenate back into single arrays
	elmo_train_new = np.concatenate(elmo_train, axis = 0)

	print("saving to file...")
	# save elmo_train_new
	pickle_out = open('elmo_train_' + mood + '.pickle','wb')
	pickle.dump(elmo_train_new, pickle_out)
	pickle_out.close()
	print("done.")
	
write_elmo_vectors('grateful')
write_elmo_vectors('happy')
write_elmo_vectors('hopeful')
write_elmo_vectors('determined')
write_elmo_vectors('aware')
write_elmo_vectors('stable')
write_elmo_vectors('frustrated')
write_elmo_vectors('overwhelmed')
write_elmo_vectors('angry')
write_elmo_vectors('guilty')
write_elmo_vectors('lonely')
write_elmo_vectors('scared')
write_elmo_vectors('sad')
