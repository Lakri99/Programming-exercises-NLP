## Excercise 6 for [SNLP 2020](https://www.lsv.uni-saarland.de/statistical-natural-language-processing-summer-2020/)

# Team Members
Lakshmi Bashyam - 2581455
Awantee Deshpande - 2581348

# Description
Solutions for programming assignment 6 of Statistical Natural Language Processing class 2020.

# Installation 
Python 3.5 or above

# Execution requirement
The .py file has to be placed in the first level of the Exercise folder, i.e, the dataset folder containing the corpora has to be in the same level as the .py file (this is in source_code directory)

#Information about the code

1)SNLPAssg6_1_1_2_1.py : Code for the Long-Range Dependencies Q1 and Language Models Q1

correlation(w1, w2, D) - calculates correlation between words w1, w2 over distance D.

find_interpolated_prob - finds interpolated probability values over the corpus

2) AbsoluteDiscounting.py : Code for finding absolute discounted probabilites

get_params(d, bigrams, unigrams, vocab) - calculates the following values over the given ngrams and vocabularuy:
	a)P_unif - uniform probability of unigrams
	b)lambdadot - eqn 11
	c)N1_plus_wi1 - eqn 10
	d)N_bigrams - dictionary of bigrams with history wi
	e)lambda_wi - eqn 9
	f) P_abs_wi - eqn 8

get_P_abs_wi1 - finds probability of bigram (wi-1, wi) with backing off conditions

find_perplexity - finds perplexity score for the bigram model

3) Kneser_Ney.py : Code for Kneser-Ney Smoothing

# To run any file
>> python filename.py