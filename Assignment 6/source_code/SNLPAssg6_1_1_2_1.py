#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import re
import operator

from collections import Counter

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')



def tokenize(text):
    #"List all the word tokens (consecutive letters) in a text. Normalize to lowercase."
    return re.findall('[a-z]+', text.lower())

# Part 1

filename = './ex6_materials/ex6_materials/English_train.txt'
with open(filename, encoding="utf-8") as f:
    text = f.read()
f.close()

tokenized_text = tokenize(text)



def correlation(w1, w2, D):
    w1_indices = [i for i, x in enumerate(tokenized_text) if x == w1]
    w2_indices = [i for i, x in enumerate(tokenized_text) if x == w2]
    
    ND_w1w2 = 0
    N = len(tokenized_text)
    
    for index in w1_indices:
        if (index + D) in w2_indices:
            ND_w1w2 += 1
    
    PD_w1w2 = ND_w1w2/N
    
    w1prob = len(w1_indices)/N
    w2prob = len(w2_indices)/N
    
    corr = PD_w1w2/(w1prob * w2prob)
    
    return corr

def find_correlation(str1, str2):
    corrs = []
    D_range = list(range(1,51))
    for D in D_range:
        corrs.append(correlation(str1, str2, D))
    
    plt.plot(D_range, corrs)
    plt.title("Correlation for words '" + str1 + "' '" + str2 +"'")
    plt.show()


find_correlation("he","his")
find_correlation("he","her")
find_correlation("she","her")
find_correlation("she","his")


# Part 2

def find_bigrams(text):
    bi_dict = {}
    for i in range(0, len(text) - 1):
        (first, second) = (text[i], text[i+1])
        if not (first, second) in bi_dict:
            bi_dict[(first, second)] = 1
        else:
            bi_dict[(first, second)] += 1
    return bi_dict

def find_unigrams(text):
    return Counter(text)

def find_interpolated_prob(bgrams, ugrams, vocab_size, alpha, lambda1, lambda2):
    inter_prob_dict = {}
    bprob_dict = {}
    uprob_dict = {}
    
    N_uni = sum(ugrams.values())
    
    for k,v in bgrams.items():
        bprob_dict[k] = (v + alpha) /(ugrams[k[0]] + (alpha * vocab_size))
    
    for k,v in ugrams.items():
        uprob_dict[k] = (v + alpha)/(N_uni + (alpha * vocab_size))
    
    for bigram in bprob_dict.keys():
        inter_prob_dict[bigram] = (lambda1 * uprob_dict[bigram[0]]) + (lambda2 * bprob_dict[bigram])
    
    return uprob_dict, bprob_dict, inter_prob_dict

def find_rel_freq(ngrams, ugrams):
    prob_dict = {}
    s = sum(ngrams.values())
    l = len(ngrams)
    for k,v in ngrams.items():
        prob_dict[k] = v / s
    return prob_dict

def find_perplexity(rel_freq, cond_prob, vocab_size):
    tsum = 0
    for key in rel_freq.keys():
        tsum -= rel_freq[key] * math.log(cond_prob[key])
    return math.exp(tsum)


trainfile = './ex6_materials/ex6_materials/English_train.txt'
with open(trainfile, encoding="utf-8") as f:
    traintext = f.read()
f.close()

testfile = './ex6_materials/ex6_materials/English_test.txt'
with open(testfile, encoding="utf-8") as f:
    testtext = f.read()
f.close()

train_text = tokenize(traintext)
test_text = tokenize(testtext)

uni_train = find_unigrams(train_text)
bi_train = find_bigrams(train_text)
vocab_size = len(uni_train)

uni_test = find_unigrams(test_text)
bi_test = find_bigrams(test_text)

for key in uni_test.keys():
    uni_train[key] = uni_train.get(key,0)

for key in bi_test.keys():
    bi_train[key] = bi_train.get(key,0)

uni_train_probs, bi_train_probs, inter_train_probs = find_interpolated_prob(bi_train, uni_train, vocab_size, 0.3, 0.5, 0.5)
uni_test_probs, bi_test_probs, inter_test_probs = find_interpolated_prob(bi_test, uni_test, vocab_size, 0.3, 0.5, 0.5)

bi_test_freq = find_rel_freq(bi_test, uni_test)
perp_bi = find_perplexity(bi_test_freq, bi_train_probs, vocab_size)
perp_inter = find_perplexity(bi_test_freq, inter_train_probs, vocab_size)



print("Bigram perplexity without interpolation = " + str(perp_bi))
print("Bigram perplexity with interpolation = " + str(perp_inter))

