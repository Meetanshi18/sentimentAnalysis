import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords, state_union
from nltk.stem import PorterStemmer


##example_text1 = "Hello Mr. Smith, how are you doing today? The weather is great and Python in awesome."
##
##print(sent_tokenize(example_text1))
##
##print(word_tokenize(example_text1))

##for i in word_tokenize(example_text):
##    print(i)

##-----------------------------------------------------------------------

##example_text2 = "This is an example showing off stop word filtration."

## words that add no meaning to the sentence
##stop_words = set(stopwords.words("english"))
##
##print("STOP-WORDS: ", stop_words)

##
##words = word_tokenize(example_text2)
##
##filtered_sentence = []
##
##for w in words:
##    if w not in stop_words:
##        filtered_sentence.append(w)
##
##print(filtered_sentence)

##------------------------------------------------------------------------

##ps = PorterStemmer()
##
##example_words = ["python","pythoning","puthoned","pythoner","pythonly"]
##
##for w in example_words:
##    print(ps.stem(w))
##
##example_text3 = "It is very important to be pythonly while you are pythoning with python. All pythoners have pythoned poorly atleast once."
##
##words = word_tokenize(example_text3)
##
##for w in words:
##    print(ps.stem(w))

##-------------------------------------------------------------------------

##
##train_text = state_union.raw("2005-GWBush.txt")
##sample_text = state_union.raw("2006-GWBush.txt")
##
##custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
##tokenized = custom_sent_tokenizer.tokenize(sample_text)

'''
Alphabetical list of part-of-speech tags used in the Penn Treebank Project:
Number
Tag
Description
1.	CC	Coordinating conjunction
2.	CD	Cardinal number
3.	DT	Determiner
4.	EX	Existential there
5.	FW	Foreign word
6.	IN	Preposition or subordinating conjunction
7.	JJ	Adjective
8.	JJR	Adjective, comparative
9.	JJS	Adjective, superlative
10.	LS	List item marker
11.	MD	Modal
12.	NN	Noun, singular or mass
13.	NNS	Noun, plural
14.	NNP	Proper noun, singular
15.	NNPS	Proper noun, plural
16.	PDT	Predeterminer
17.	POS	Possessive ending
18.	PRP	Personal pronoun
19.	PRP$	Possessive pronoun
20.	RB	Adverb
21.	RBR	Adverb, comparative
22.	RBS	Adverb, superlative
23.	RP	Particle
24.	SYM	Symbol
25.	TO	to
26.	UH	Interjection
27.	VB	Verb, base form
28.	VBD	Verb, past tense
29.	VBG	Verb, gerund or present participle
30.	VBN	Verb, past participle
31.	VBP	Verb, non-3rd person singular present
32.	VBZ	Verb, 3rd person singular present
33.	WDT	Wh-determiner
34.	WP	Wh-pronoun
35.	WP$	Possessive wh-pronoun
36.	WRB	Wh-adverb

'''
##
##print(tokenized)
####
##def process_content():
##    try:
##        for i in tokenized:
##            words = nltk.word_tokenize(i)
##            # pos tagging only works with tokenized sentences, not complete paragraphs
##            tagged = nltk.pos_tag(words)
##            print(tagged)
##
##            
##    except Exception as e:
##        print(str(e))
##        
##process_content()

##-------------------------------------------------------------------------

sentence = "This is fun. Modi should be elected again. Parties are boring without friends."
words = nltk.word_tokenize(sentence)
tagged = nltk.pos_tag(words)
print(tagged)


chunkGram = r"""Chunk: {<.*>+}
                       }<VB.?|IN|DT|TO>+{"""

chunkParser = nltk.RegexpParser(chunkGram)
chunked = chunkParser.parse(tagged)
chunked.draw()
##-------------------------------------------------------------------------

## 7, 8, 9 to be done

print(nltk.__file__)























