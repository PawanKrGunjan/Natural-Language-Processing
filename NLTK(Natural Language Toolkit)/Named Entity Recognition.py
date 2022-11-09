import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw('2005-GWBush.txt')
sample_text = state_union.raw('2006-GWBush.txt')

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

Parts_of_Speech = {
    'CC'  : ' coordinating conjunction', 
    'CD'  : ' cardinal digit', 
    'DT'  : ' determiner', 
    'EX'  : ' existential there (like: "there is" ... think of it like "there exists")', 
    'FW'  : ' foreign word', 
    'IN'  : ' preposition/subordinating conjunction', 
    'JJ'  : " adjective 'big'", 
    'JJR' : " adjective, comparative 'bigger'", 
    'JJS' : " adjective, superlative 'biggest'", 
    'LS'  : ' list marker 1)', 
    'MD'  : ' modal could, will', 
    'NN'  : " noun, singular 'desk'", 
    'NNS' : " noun plural 'desks'",
    'NNP' : " proper noun, singular 'Pawan'", 
    'NNPS': " proper noun, plural 'Boys'", 
    'PDT' : " predeterminer 'all the kids'", 
    'POS' : " possessive ending parent's", 
    'PRP' : ' personal pronoun I, he, she', 
    'PRP$': ' possessive pronoun my, his, hers', 
    'RB'  : ' adverb very, silently,', 
    'RBR' : ' adverb, comparative better', 
    'RBS' : ' adverb, superlative best', 
    'RP'  : ' particle give up', 
    'TO'  : " to go 'to' the store.", 
    'UH'  : ' interjection errrrrrrrm', 
    'VB'  : ' verb, base form take', 
    'VBD' : ' verb, past tense took', 
    'VBG' : ' verb, gerund/present participle taking', 
    'VBN' : ' verb, past participle taken', 
    'VBP' : ' verb, sing. present, non-3d take', 
    'VBZ' : ' verb, 3rd person sing. present takes', 
    'WDT' : ' wh-determiner which', 
    'WP'  : ' wh-pronoun who, what', 
    'WP$' : ' possessive wh-pronoun whose', 
    'WRB' : ' wh-abverb where, when'
    }

"""
NE Type Examples
ORGANIZATION : Georgia-Pacific Corp., WHO
PERSON : Eddy Bonte, President Kalam
LOCATION : Delhi
DATE : January, 23-01-1996
TIME : 1:15 p.m., two fifty a m
MONEY : 12 LAKH INDIAN RUPEES
PERCENT : 18.75%
FACILITY : Washington Monument,
GPE :South East Asia, Midlothian
"""
# print(Parts_of_Speech)

def process_content():
    try:
        for i in tokenized[:5]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            namedEnt = nltk.ne_chunk(tagged, binary = True)
            namedEnt.draw()

    except Exception as e:
        print(str(e))

process_content()