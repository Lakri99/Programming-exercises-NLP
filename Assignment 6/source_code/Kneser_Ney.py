import math
import re
# import operator

from collections import Counter

def tokenize(text):
    "List all the word tokens (consecutive letters) in a text. Normalize to lowercase."
    return re.findall('[a-z]+', text.lower())

def get_grams(word_list,gram):
    if gram == 2:
        gram_list = [(word_list[i-1],word_list[i]) for i in range(1,len(word_list))]
    elif gram == 1:
        gram_list = word_list
    gram_counter = Counter(gram_list)
    return gram_counter

def condn_gram(gram_dict, word, pos):
    counter = 0
    word_at_index =""
    for word_set in gram_dict.keys():
        if pos == "start":
            word_at_index = word_set[0]
        elif pos == "end":
            word_at_index = word_set[1]
        counter += 1 if word_at_index == word else 0
    return counter


train_filename = './ex6_materials/English_train.txt'
with open(train_filename) as f:
    train = f.read()
f.close()
train_filename = './ex6_materials/English_test.txt'
with open(train_filename) as f:
    test = f.read()
f.close()

train = tokenize(train)
test = tokenize(test)
bigram_train = get_grams(train, 2)
print(Counter(train).get("pleasure"))
unigram_train = get_grams(train, 1)
alpha = 0.3

print("-------------------longbourn----------------")
w = "longbourn"
print("N(w)",unigram_train.get(w,-1))
print("N(o - W)", condn_gram(bigram_train, w, "end"))
kn_w3 = condn_gram(bigram_train, w, "end") / len(bigram_train)
print("PKN(w)  ",-math.log(kn_w3,2))
lid = (unigram_train[w] + alpha) / (sum(unigram_train.values()) + (alpha * len(unigram_train)))
print("PLID(w)   ",-math.log(lid,2))

print("-----------pleasure----------")
w1 = "pleasure"
print("N(w)",unigram_train.get(w1,-1))
print("N(o - W)", condn_gram(bigram_train, w1, "end"))
kn_w3 = condn_gram(bigram_train, w1, "end") / len(bigram_train)
print("PKN(w)   ",-math.log(kn_w3,2))
lid = (unigram_train[w1] + alpha) / (sum(unigram_train.values()) + (alpha * len(unigram_train)))
print("PLID(w)   ",-math.log(lid,2))


