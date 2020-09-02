#!/usr/bin/env python
# coding: utf-8

import math
import re
import operator

from collections import Counter

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


def tokenize(text):
    #"List all the word tokens (consecutive letters) in a text. Normalize to lowercase."
    return re.findall('[a-z]+', text.lower())


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

vocab = list(uni_train.keys())
V = len(vocab)

uni_test = find_unigrams(test_text)
bi_test = find_bigrams(test_text)


def get_lambda_dot(d, unigrams):
    N1_plus = len(unigrams)
    N = sum(unigrams.values())
    lambdadot = (d/N) * N1_plus
    return lambdadot

def get_params(d, bigrams, unigrams, vocab):
    
    #params to return
    N1_plus_wi1 = dict() #eqn 10
    N_bigrams = dict() #denominator of 7, 9
    lambda_wi = dict() #eqn 9
    P_abs_wi = dict() #eqn 8
    
    P_unif = 1/len(vocab)
    N_unis = sum(unigrams.values())
    lambdadot = get_lambda_dot(d, unigrams)
    
    for word in vocab:
        count = 0 #unique bigrams starting with word
        N = 0 #total number of bigrams starting with word
        
        for key, val in bigrams.items():
            if(key[0] == word):
                if(val > 0):
                    count += 1
                    N += val
        
        N1_plus_wi1[word] = count 
        N_bigrams[word] = N
        
        if N > 0:
            lambdad = (d/N) * N1_plus_wi1[word]
        else:
            lambdad = 0
            
        lambda_wi[word] = lambdad
        P_abs_wi[word] = (max((unigrams[word] - d), 0)/N_unis) + (lambdadot * P_unif)
          
    return P_unif, lambdadot, N1_plus_wi1, N_bigrams, lambda_wi, P_abs_wi


def get_P_abs_wi1(wi, wi_1, d, unigrams, bigrams, N_bigrams, lambda_wi, P_abs_wi, lambdadot, P_unif):
    try:
        if((wi_1, wi) not in bigrams.keys()):
            if(wi_1 not in lambda_wi.keys()):
                if(wi not in P_abs_wi.keys()):
                    P_abs_wi1 = lambdadot * P_unif
                else:
                    P_abs_wi1 = P_abs_wi[wi]
            else:
                P_abs_wi1 = (lambda_wi[wi_1] * P_abs_wi[wi])
        else:
            N = N_bigrams[wi_1]
            P_abs_wi1 = (max((bigrams[(wi_1, wi)] - d), 0)/N) + (lambda_wi[wi_1] * P_abs_wi[wi])
    except:
        P_abs_wi1 = lambdadot * P_unif
    return P_abs_wi1


def find_perplexity(bgrams, bigrams, ugrams, vocab, d):
    rel_dict = {}
    cond_dict = {}
    tsum = 0
    
    P_unif, lambdadot, N1_plus_wi1, N_bigrams, lambda_wi, P_abs_wi = get_params(d, bigrams, ugrams, vocab)
    
    s = sum(bgrams.values())
    l = len(bgrams)
    
    for k,v in bgrams.items():
        rel_dict[k] = v / s
        cond_dict[k] = get_P_abs_wi1(k[1], k[0], d, ugrams, bigrams, N_bigrams, lambda_wi, P_abs_wi, lambdadot, P_unif)
        tsum -= rel_dict[k] * math.log(cond_dict[k])
    
    perplexity = math.exp(tsum)
    return rel_dict, cond_dict, perplexity


# Run test cases
unigrams = uni_train
vocab = list(uni_train.keys())
bigrams = bi_train

P_unif, lambdadot, N1_plus_wi1, N_bigrams, lambda_wi, P_abs_wi = get_params(0.7, bigrams, unigrams, vocab)
precision = 10**-8
                 
histories = ['the', 'in', 'at', 'blue', 'white']

for h in histories:
    print("History = " + h)
    P_sum = sum(get_P_abs_wi1(w, h, 0.7, unigrams, bigrams, N_bigrams, lambda_wi, P_abs_wi, lambdadot, P_unif) for w in vocab)
    print(f"P_sum = {P_sum} \n")
    assert abs(1.0 - P_sum) < precision, 'Probability mass does not sum up to one for history' + h

print('TEST SUCCESSFUL!')


# Find perplexity for d = 0.7

rel_dict, cond_dict, pp = find_perplexity(bi_test, bigrams, unigrams, vocab, 0.7)
print(pp)


# Find perplexity for given range of d values

D = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
print(D)
pp_d = []

for d in D:
    rel_dict, cond_dict, pp = find_perplexity(bi_test, bigrams, unigrams, vocab, d)
    pp_d.append(pp)


plt.plot(D, pp_d)
plt.xlabel("discounting parameter d")
plt.ylabel("perplexity")
plt.title("Perplexity vs Discounting parameter")
plt.show()

print(pp_d)