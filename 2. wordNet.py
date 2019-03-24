from nltk.corpus import wordnet

##syns = wordnet.synsets("program")
##
### synset
##print(syns[:5])
##print(syns[0].name())
##
##print(syns[0].lemmas())
### just the word
##print(syns[0].lemmas()[0].name())
##
### definition
##print(syns[0].definition())
##
### examples
##print(syns[0].examples())
##print('')
##
synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
    print(syn)
    for l in syn.lemmas():
        print(l)
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(synonyms)
print('')
print(antonyms)

#-----------------------------------------------------------------

##w1 = wordnet.synset("ship.n.01")
##w2 = wordnet.synset("boat.n.01")
### prints the similarity between the two words
##print(w1.wup_similarity(w2))
