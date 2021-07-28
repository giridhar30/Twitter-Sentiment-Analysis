
import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import word_tokenize
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC

short_pos = open('short_reviews/positive.txt').read()
short_neg = open('short_reviews/negative.txt').read()

docs = [(r, 'pos') for r in short_pos.split('\n')]

for r in short_neg.split('\n'):
	docs.append((r, 'neg'))

save_docs = open('docs.pickle', 'wb')
pickle.dump(docs, save_docs)
save_docs.close()

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

all_words = []

for w in short_pos_words:
	all_words.append(w.lower())

for w in short_neg_words:
	all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

save_all_words = open('all_words.pickle', 'wb')
pickle.dump(all_words, save_all_words)
save_all_words.close()

print(all_words['tarantino'])

word_features = list(all_words.keys())[:5000]

save_word_features = open('word_features.pickle', 'wb')
pickle.dump(word_features, save_word_features)
save_word_features.close()

def find_features(doc):
	words = word_tokenize(doc)
	features = {}
	for w in word_features:
		features[w] = (w in words)
	return features

featuresets = [(find_features(rev), category) for (rev, category) in docs]
random.shuffle(featuresets)

save_featuresets = open('featuresets.pickle', 'wb')
pickle.dump(featuresets, save_featuresets)
save_featuresets.close()

# +ve dataset
train = featuresets[:10000]
test = featuresets[10000:]

# -ve dataset
train = featuresets[100:]
test = featuresets[:100]

classifier = nltk.NaiveBayesClassifier.train(train)

print('Org NB Accuracy%:', nltk.classify.accuracy(classifier, test)*100)
classifier.show_most_informative_features(20)

save_classifier = open('naivebayes_org.pickle', 'wb')
pickle.dump(classifier, save_classifier)
save_classifier.close()

# multinomial Naive Bayes
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(train)
print('Multinomial NB Accuracy%:', nltk.classify.accuracy(MNB_classifier, test)*100)

save_MNB_classifier = open('MNB_classifier.pickle', 'wb')
pickle.dump(MNB_classifier, save_MNB_classifier)
save_MNB_classifier.close()

# Bernoulli Naive Bayes
BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(train)
print('Bernoulli NB Accuracy%:', nltk.classify.accuracy(BNB_classifier, test)*100)

save_BNB_classifier = open('BNB_classifier.pickle', 'wb')
pickle.dump(BNB_classifier, save_BNB_classifier)
save_BNB_classifier.close()

# LogisticRegression, SGDClassifier, LinearSVC

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(train)
print('LogisticRegression Accuracy%:', nltk.classify.accuracy(LogisticRegression_classifier, test)*100)

save_LogisticRegression_classifier = open('LogisticRegression_classifier.pickle', 'wb')
pickle.dump(LogisticRegression_classifier, save_LogisticRegression_classifier)
save_LogisticRegression_classifier.close()

SGD_classifier = SklearnClassifier(SGDClassifier())
SGD_classifier.train(train)
print('SGD classifier Accuracy%:', nltk.classify.accuracy(SGD_classifier, test)*100)

save_SGD_classifier = open('SGD_classifier.pickle', 'wb')
pickle.dump(SGD_classifier, save_SGD_classifier)
save_SGD_classifier.close()

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(train)
print('LinearSVC classifier Accuracy%:', nltk.classify.accuracy(LinearSVC_classifier, test)*100)

save_LinearSVC_classifier = open('LinearSVC_classifier.pickle', 'wb')
pickle.dump(LinearSVC_classifier, save_LinearSVC_classifier)
save_LinearSVC_classifier.close()
