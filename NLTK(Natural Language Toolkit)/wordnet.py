from nltk.corpus import wordnet

# word = 'program'
word = 'good'

syns = wordnet.synsets(word)
print('*'*25,'Wordnet Synsets','*'*25)
print(syns)

# synset
print('*'*25,'Synset Name','*'*25)
for syn in wordnet.synsets(word):
    print('Name-->',syn.name())
    # definition
    print('Definition >>',syn.definition())
    # examples
    print('Examples >>',syn.examples())
    print('-'*100)
#print(syns[0].name())

# Just the word
print('*'*25,'Just the word','*'*25)
for syn in wordnet.synsets(word):
    for l in syn.lemmas():
        print('Lemma -->',l.name())
#print(syns[0].lemmas()[0].name())
print('*'*100)

synonyms = []
antonyms = []

for syn in wordnet.synsets(word):
    for l in syn.lemmas():
        #print('Lemma -->',l)
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print('Synonyms >>>',set(synonyms))
print('*'*100)
print('Antonyms >>>',set(antonyms))

# Check word Similarity 
print('*'*25,'Word Similarity Score','*'*25)
w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('boat.n.01')
print('Similarity Score ship & boat >>',w1.wup_similarity(w2))

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('aeroplane.n.01')
print(f'Similarity Score between ship & aeroplane >>',w1.wup_similarity(w2))

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('car.n.01')
print(f'Similarity Score between ship & car >>',w1.wup_similarity(w2))

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('cat.n.01')
print(f'Similarity Score between ship & cat >>',w1.wup_similarity(w2))