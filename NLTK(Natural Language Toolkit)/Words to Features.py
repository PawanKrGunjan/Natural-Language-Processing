import nltk
import random
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(file_id)), category) 
                for category in movie_reviews.categories()
                for file_id in movie_reviews.fileids(category)]

random.shuffle(documents)

#print('Documnet Length',len(documents))
#print(documents[0])

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

Frequency = nltk.FreqDist(all_words)

print('-'*25,'20 Most common words','-'*25)
print(Frequency.most_common(20))
print('Good word frequency',Frequency["good"])
print('Stupid word frequency',Frequency["stupid"])

word_features = list(Frequency.keys())[:500]
#print(word_features)

def find_features(documents):
    words = set(documents)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

feature_sets = [(find_features(rev), category) for (rev,category) in documents]