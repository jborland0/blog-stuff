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
from joblib import dump

def scikit_mood(mood):
	# read data
	train = pd.read_csv('elmo/train_' + mood + '.csv')

	# load elmo_train_new
	pickle_in = open('elmo/elmo_train_' + mood + '.pickle', 'rb')
	elmo_train_new = pickle.load(pickle_in)

	# split into training and validation
	xtrain, xvalid, ytrain, yvalid = train_test_split(elmo_train_new, 
													  train['label'],  
													  random_state=42, 
													  test_size=0.2)

	# create and train classification model
	lreg = LogisticRegression()
	lreg.fit(xtrain, ytrain)

	# save model to file
	dump(lreg, 'elmo/scikit_' + mood + '.joblib')
	print('wrote scikit_' + mood + '.joblib')
	
scikit_mood('grateful')
scikit_mood('happy')
scikit_mood('hopeful')
scikit_mood('determined')
scikit_mood('aware')
scikit_mood('stable')
scikit_mood('frustrated')
scikit_mood('overwhelmed')
scikit_mood('angry')
scikit_mood('guilty')
scikit_mood('lonely')
scikit_mood('scared')
scikit_mood('sad')
