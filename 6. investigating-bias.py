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

# WORDS AS FEATURES FOR LEARNING

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# random.shuffle(documents)

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

# a list that contains words as keys from most frequency to least
# along with the the frequency
all_words = nltk.FreqDist(all_words)

# list of 3000 most common words
word_features = list(all_words.keys())[:3000]
# print(word_features)

def find_features(document):
    # document is a collection of words as we saw above
    # set so that a particular word comes only once
    words = set(document)
    features = {}
    for w in word_features:
        # checking which features exist in our document
        features[w] = (w in words)

    return features

# print(find_features(movie_reviews.words('neg/cv000_29416.txt')))

featuresets = [(find_features(rev), category) for (rev, category) in documents]
# print(featuresets[:1])

# positive data
training_set = featuresets[:1900]
testing_set = featuresets[1900:]

# negative data
training_set = featuresets[100:]
testing_set = featuresets[:100]

# posterior = prior occurrences x liklihood / evidence

# classifier = nltk.NaiveBayesClassifier.train(training_set)

# -------ACCESSING SAVED ITEMS-------------------------------------

classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

# -----------------------------------------------------------------



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















  


