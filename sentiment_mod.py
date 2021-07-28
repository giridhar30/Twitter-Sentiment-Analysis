
import nltk
from nltk.classify import ClassifierI
from nltk.tokenize import word_tokenize
import pickle
from statistics import mode

class VoteClassifier(ClassifierI):
	def __init__(self, *classifiers):
		self._classifiers = classifiers

	def classify(self, features):
		return mode([c.classify(features) for c in self._classifiers])

	def confidence(self, features):
		votes = [c.classify(features) for c in self._classifiers]
		choice_votes = votes.count(mode(votes))
		conf = choice_votes / len(votes)
		return conf

word_features_file = open('word_features.pickle', 'rb')
word_features = pickle.load(word_features_file)
word_features_file.close()

def find_features(doc):
	words = word_tokenize(doc)
	features = {}
	for w in word_features:
		features[w] = (w in words)
	return features

classifier_file = open('naivebayes_org.pickle', 'rb')
classifier = pickle.load(classifier_file)
classifier_file.close()

MNB_classifier_file = open('MNB_classifier.pickle', 'rb')
MNB_classifier = pickle.load(MNB_classifier_file)
MNB_classifier_file.close()

BNB_classifier_file = open('BNB_classifier.pickle', 'rb')
BNB_classifier = pickle.load(BNB_classifier_file)
BNB_classifier_file.close()

LogisticRegression_classifier_file = open('LogisticRegression_classifier.pickle', 'rb')
LogisticRegression_classifier = pickle.load(LogisticRegression_classifier_file)
LogisticRegression_classifier_file.close()

SGD_classifier_file = open('SGD_classifier.pickle', 'rb')
SGD_classifier = pickle.load(SGD_classifier_file)
SGD_classifier_file.close()

LinearSVC_classifier_file = open('LinearSVC_classifier.pickle', 'rb')
LinearSVC_classifier = pickle.load(LinearSVC_classifier_file)
LinearSVC_classifier_file.close()

voted_classifier = VoteClassifier(
	classifier,
	MNB_classifier, 
	BNB_classifier, 
	LogisticRegression_classifier, 
	SGD_classifier, 
	LinearSVC_classifier)

def sentiment(txt):
	feats = find_features(txt)
	return voted_classifier.classify(feats), voted_classifier.confidence(feats)
