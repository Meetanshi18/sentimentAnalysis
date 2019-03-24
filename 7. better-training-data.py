import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
# MultinomialNB : multinomial not binary distribution i.e. many categories not 2

# importing different algorithms so that we'll put a review in the category found by majority of the algorithms
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

short_pos = open("short_reviews/positive.txt", "r").read()
short_neg = open("short_reviews/negative.txt", "r").read()

documents = []

for r in short_pos.split('\n'):
    documents.append((r, "pos"))

for r in short_neg.split('\n'):
    documents.append((r, "neg"))

all_words = []

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for w in short_pos_words:
    all_words.append(w.lower())

for w in short_neg_words:
    all_words.append(w.lower())

# a list that contains words as keys from most frequency to least
# along with the the frequency
all_words = nltk.FreqDist(all_words)

# list of 3000 most common words
word_features = list(all_words.keys())[:3000]
# print(word_features)

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        # checking which features exist in our document
        features[w] = (w in words)

    return features

# print(find_features(movie_reviews.words('neg/cv000_29416.txt')))

featuresets = [(find_features(rev), category) for (rev, category) in documents]
# print(featuresets[:1])

random.shuffle(featuresets)

training_set = featuresets[:2000]
testing_set = featuresets[2000:4000]


# posterior = prior occurrences x liklihood / evidence

classifier = nltk.NaiveBayesClassifier.train(training_set)

print("Original Naive Bayes Algo accuracy percent: ", (nltk.classify.accuracy(classifier, testing_set))*100)
# classifier.show_most_informative_features(15)

# ------------------------------------------------------------------------------------

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_Classifier Algo accuracy percent: ", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier Algo accuracy percent: ", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier Algo accuracy percent: ", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier Algo accuracy percent: ", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC_classifier Algo accuracy percent: ", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier Algo accuracy percent: ", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier Algo accuracy percent: ", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

# ----------------------VOTED CLASSIFIER------------------------------------

voted_classifier = VoteClassifier(MNB_classifier, BernoulliNB_classifier, LogisticRegression_classifier, SGDClassifier_classifier, SVC_classifier, LinearSVC_classifier, NuSVC_classifier)
print("voted_classifier accuracy percent: ", (nltk.classify.accuracy(voted_classifier, testing_set))*100)















  


