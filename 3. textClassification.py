import nltk
import random
from nltk.corpus import movie_reviews

documents = []
# dont run this. it doesnt work. below one liner works
##print(movie_reviews.categories())
##for category in movie_reviews.categories():
##    print(category)
##    for fileid in movie_reviews.fileids(category):
##        print(fileid)
##        print(list(movie_reviews.words(fileid)), category)
##        # appending the list of words of a particular review along with its category in documents
##        documents.append(list(movie_reviews.words(fileid)), category)


# understand the above commented block to understand whats happening in this one liner
##documents = [(list(movie_reviews.words(fileid)), category)
##             for category in movie_reviews.categories()
##             for fileid in movie_reviews.fileids(category)]
##
##random.shuffle(documents)
##print(documents[1])

#-----------------------------------------------------------------
##all_words = []
##for w in movie_reviews.words():
##    all_words.append(w.lower())
##
##all_words = nltk.FreqDist(all_words)

##print(all_words.most_common(5))
##print(all_words["awesome"])

# -----------------------------------------------------------------
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

word_features = list(all_words.keys())[:3000]
##print(word_features)

def find_features(review_words):
    # set so that a particular word comes only once
    words = set(review_words)
    features = {}
    for w in word_features:
        # checking which features exist in our document
        features[w] = {w in words}

    return features

##print(find_features(movie_reviews.words('neg/cv000_29416.txt')))

# set of words that are present in a particular review, also are part of top 3000 words of all reviews, along with the review's category - is one entry
feature_sets = [(find_features(rev), category) for (rev, category) in documents]
print(feature_sets[:1])


























