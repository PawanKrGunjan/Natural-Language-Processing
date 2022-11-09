from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


print(lemmatizer.lemmatize("Cats"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("rocks"))
print(lemmatizer.lemmatize("Python"))
print(lemmatizer.lemmatize("geeks"))

print('-'*25)
print(lemmatizer.lemmatize("better", pos ='a'))
print(lemmatizer.lemmatize("best", pos ='a'))
print(lemmatizer.lemmatize("runs"))
print(lemmatizer.lemmatize("runs", pos ='a'))
print(lemmatizer.lemmatize("runs", pos ='n'))