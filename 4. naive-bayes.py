import nltk
import random
from nltk.corpus import movie_reviews
import pickle

# WORDS AS FEATURES FOR LEARNING

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

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

training_set = featuresets[:1900]
testing_set = featuresets[1900:]

# classifier = nltk.NaiveBayesClassifier.train(training_set)

# -------ACCESSING SAVED CLASSIFIER SO THAT WE DONT HAVE TO TRAIN IT AGAIN AND AGAIN-------


classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

# -----------------------------------------------------------------



# print("Naive Bayes Algo accuracy percent: ", (nltk.classify.accuracy(classifier, testing_set))*100)

classifier.show_most_informative_features(15)


# -----------------------SAVING THE TRAINED CLASSIFIER--------------------------------------
##save_classifier = open("naivebayes.pickle", "wb")
##pickle.dump(classifier, save_classifier)
##save_classifier.close()
# -----------------------------------------------------------------





























