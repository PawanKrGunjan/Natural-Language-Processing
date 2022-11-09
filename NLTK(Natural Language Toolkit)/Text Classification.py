import nltk
import random
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(file_id)), category) 
                for category in movie_reviews.categories()
                for file_id in movie_reviews.fileids(category)]

random.shuffle(documents)

print('Documnet Length',len(documents))
#print(documents[0])

Positive = []
Negative = []
for category in movie_reviews.categories():
    print(category)
    if category =='pos':
        Positive.append(list(movie_reviews.words(file_id) for file_id in movie_reviews.fileids('pos')))
    else:
        Negative.append(list(movie_reviews.words(file_id) for file_id in movie_reviews.fileids('neg')))

print('Postive >>',Positive)
print('Negative >>',Negative)

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

Frequency = nltk.FreqDist(all_words)

print('-'*25,'20 Most common words','-'*25)
print(Frequency.most_common(20))
print('Good word frequency',Frequency["good"])
print('Stupid word frequency',Frequency["stupid"])