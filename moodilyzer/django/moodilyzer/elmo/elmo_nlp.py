import tensorflow_hub as hub
import tensorflow.compat.v1 as tf
import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm
import re
import os
import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from joblib import load

class ElmoClassifier:
	def __init__(self, pickle_dir, text):
		pd.set_option('display.max_colwidth', 200)
		self.pickle_dir = pickle_dir

		#To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
		tf.disable_eager_execution()

		# create elmo
		print("loading elmo...")
		self.elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
		print("elmo loaded.")

		# import spaCy's language model
		self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
		
		# create elmo vectors for input text
		self.elmo_text = self.create_elmo_vectors(text)

	# function to lemmatize text
	def lemmatization(self, texts):
		output = []
		for i in texts:
			s = [token.lemma_ for token in self.nlp(i)]
			output.append(' '.join(s))
		return output

	# function to return average elmo vector for each sentence in x
	def elmo_vectors(self, x):
		embeddings = self.elmo(x.tolist(), signature="default", as_dict=True)["elmo"]

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			sess.run(tf.tables_initializer())
			# return average of ELMo features
			return sess.run(tf.reduce_mean(embeddings,1))

	def create_elmo_vectors(self, text):
		# create fake tweet file
		data = [[0, text]] 
	  
		# Create the pandas DataFrame 
		train = pd.DataFrame(data, columns = ['id', 'tweet'])

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
		train['clean_tweet'] = self.lemmatization(train['clean_tweet'])

		# chop up sentences into batches of 100
		list_train = [train[i:i+1] for i in range(0,train.shape[0],100)]

		# Extract ELMo embeddings
		print("creating elmo training vectors...")
		elmo_train = [self.elmo_vectors(x['clean_tweet']) for x in list_train]

		# concatenate back into single arrays
		elmo_train_new = np.concatenate(elmo_train, axis = 0)
		
		# return result
		return elmo_train_new

	def classify_mood(self, mood):

		# load classification model
		lreg = load(os.path.join(self.pickle_dir, 'scikit_' + mood + '.joblib'))

		# classify!
		classification = lreg.predict(self.elmo_text)
		return int(classification[0]) * 100

	def classify(self):
		print('grateful: ' + str(self.classify_mood('grateful')))
		print('happy: ' + str(self.classify_mood('happy')))
		print('hopeful: ' + str(self.classify_mood('hopeful')))
		print('determined: ' + str(self.classify_mood('determined')))
		print('aware: ' + str(self.classify_mood('aware')))
		print('stable: ' + str(self.classify_mood('stable')))
		print('frustrated: ' + str(self.classify_mood('frustrated')))
		print('overwhelmed: ' + str(self.classify_mood('overwhelmed')))
		print('angry: ' + str(self.classify_mood('angry')))
		print('guilty: ' + str(self.classify_mood('guilty')))
		print('lonely: ' + str(self.classify_mood('lonely')))
		print('scared: ' + str(self.classify_mood('scared')))
		print('sad: ' + str(self.classify_mood('sad')))
