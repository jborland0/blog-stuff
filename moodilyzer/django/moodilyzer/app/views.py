from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import json
from lstm.lstmclassifier import LSTMClassifier
from elmo.elmo_nlp import ElmoClassifier

def sample_lstm_classify(request):
	text = request.GET.get('text', '')
	lstm = LSTMClassifier('/directory/where/your/lstm/models/are/')
	return JsonResponse({'grateful':lstm.moodilyze(text,'grateful'),'happy':lstm.moodilyze(text,'happy'),'hopeful':lstm.moodilyze(text,'hopeful'),'determined':lstm.moodilyze(text,'determined'),'aware':lstm.moodilyze(text,'aware'),'stable':lstm.moodilyze(text,'stable'),'frustrated':lstm.moodilyze(text,'frustrated'),'overwhelmed':lstm.moodilyze(text,'overwhelmed'),'guilty':lstm.moodilyze(text,'guilty'),'angry':lstm.moodilyze(text,'angry'),'lonely':lstm.moodilyze(text,'lonely'),'scared':lstm.moodilyze(text,'scared'),'sad':lstm.moodilyze(text,'sad')})

def sample_elmo_classifier(request):
	text = request.GET.get('text', '')
	elmo = ElmoClassifier('/directory/where/your/elmo/models/are/')
	return JsonResponse({'grateful':elmo.classify_mood(text,'grateful'),'happy':elmo.classify_mood(text,'happy'),'hopeful':elmo.classify_mood(text,'hopeful'),'determined':elmo.classify_mood(text,'determined'),'aware':elmo.classify_mood(text,'aware'),'stable':elmo.classify_mood(text,'stable'),'frustrated':elmo.classify_mood(text,'frustrated'),'overwhelmed':elmo.classify_mood(text,'overwhelmed'),'guilty':elmo.classify_mood(text,'guilty'),'angry':elmo.classify_mood(text,'angry'),'lonely':elmo.classify_mood(text,'lonely'),'scared':elmo.classify_mood(text,'scared'),'sad':elmo.classify_mood(text,'sad')})
