from nltk.corpus import brown
from collections import Counter
import matplotlib
import math
import matplotlib.pyplot as plt
import re
import string

import nltk
# downloading data
nltk.download('brown')

def get_relative_freq(gram_dict):
    result_dict = {}
    for k,v in gram_dict.items():
        result_dict[k] = v / sum(gram_dict.values())
    return result_dict

def get_expectation(prob_dict):
    return sum([-math.log2(prob) for prob in prob_dict.values()])/len(prob_dict)

def get_bigrams(word_list):
    bi_grams = [(word_list[i-1],word_list[i]) for i in range(1,len(word_list))]
    bi_gram_counter = Counter(bi_grams)
    return bi_gram_counter

def Preprocessing(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[\u4e00-\u9fff«–\u200b„“ˈ—\ufeff]+','',text)
    text = text.lower().replace("\n"," ")
    text = text.split(' ')
    return text

# ------------------------------------------
# Problem 1
# -----------------------------------------

# Preprocess - tokenise based on whitespace
words = brown.words()

# Lowercase
words = [word.lower() for word in words]

# Calculate bigrams
bi_gram_counter = get_bigrams(words)
in_gram ={}
the_gram = {}

# Count the conditional districution

for bi_word in bi_gram_counter.keys():
    if bi_word[0] == "in":
        in_gram[bi_word] = bi_gram_counter[bi_word]
    elif bi_word[0] == "the":
        the_gram[bi_word] = bi_gram_counter[bi_word]


in_gram_prob = get_relative_freq(in_gram)
the_gram_prob = get_relative_freq(the_gram)

in_top_20 = [v for k,v in sorted(in_gram_prob.items(), key=lambda item: item[1], reverse=True)[:20]]
in_words_top_20 = [k[1] for k,v in sorted(in_gram_prob.items(), key=lambda item: item[1], reverse=True)[:20]]
the_top_20 = [v for k,v in sorted(the_gram_prob.items(), key=lambda item: item[1], reverse=True)[:20]]
the_words_top_20 = [k[1] for k,v in sorted(the_gram_prob.items(), key=lambda item: item[1], reverse=True)[:20]]

x_range = list(range(0,20))
plt.bar(x_range, list(in_top_20))
plt.ylabel("Bigram probability")
plt.xticks(x_range, in_words_top_20, rotation=45)
plt.title("Probability distribution of In-Bigrams")
plt.show()

plt.bar(x_range, list(the_top_20))
plt.ylabel("Bigram probability")
plt.xticks(x_range, the_words_top_20, rotation=45)
plt.title("Probability distribution of The-Bigrams")
plt.show()
# plt.savefig('Ranking_comp.png')

# Expectation value
print("Expectation value of in "+str(get_expectation(in_gram_prob)))
print("Expectation value of the "+str(get_expectation(the_gram_prob)))



# ------------------------------------------------
# Problem 2
# -------------------------------------------------
filename = './exercise3_materials/English_train.txt'
with open(filename,encoding='utf-8') as f:
    text_train = f.read()
f.close()

filename = './exercise3_materials/English_test.txt'
with open(filename,encoding='utf-8') as f:
    text_test = f.read()
f.close()

# Preprocessing
text_train = Preprocessing(text_train)
text_test = Preprocessing(text_test)

# Find perplexity value on test corpus
def get_pp(alpha, text_train, text_test):
    bi_gram_test = get_bigrams(text_test)
    bi_gram_train = get_bigrams(text_train)
    uni_gram_test = Counter(text_test)
    uni_gram_train = Counter(text_train)
    pp_bigram = 0
    pp_unigram = 0
    V = len(uni_gram_train)
    for key, val in bi_gram_test.items():
        # V = sum(bi_gram_train.values())
        prob = (bi_gram_train.get(key, 0) + alpha) / (uni_gram_train[key[0]] + (alpha * V))
        rel_freq = val / sum(bi_gram_test.values())
        pp_bigram -= rel_freq * math.log(prob)
    pp_bigram = math.exp(pp_bigram)
    for key, val in uni_gram_test.items():
        # V = sum(uni_gram_train.values())
        prob = (uni_gram_train.get(key, 0) + alpha) / (sum(uni_gram_train.values()) + (alpha * V))
        rel_freq = val / sum(uni_gram_test.values())
        pp_unigram -= rel_freq * math.log(prob)
    pp_unigram = math.exp(pp_unigram)
    return pp_unigram, pp_bigram

uni_perps = []
bi_perps = []

# split train into 20%....
k = math.ceil(len(text_train)/5)
sample_sizes = [k, 2 * k, 3 * k, 4 * k, 5*k]
for size in sample_sizes:
    res = text_train[:size]
    output = get_pp(0.03, res, text_test)
    uni_perps.append(output[0])
    bi_perps.append(output[1])

print(bi_perps, uni_perps)
plt.plot(range(0, len(sample_sizes)), uni_perps)
plt.ylabel("Perplexity")
plt.xlabel("Size of corpus")
plt.xticks(range(0, len(sample_sizes)),["20%", "40%", "60%", "80%", "100%"])
plt.title("Change in Unigram Perplexity")
plt.show()
plt.plot(range(0, len(sample_sizes)), bi_perps)
plt.ylabel("Perplexity")
plt.xlabel("Size of corpus")
plt.xticks(range(0, len(sample_sizes)),["20%", "40%", "60%", "80%", "100%"])
plt.title("Change in Bigram Perplexity")
plt.show()

# ------------------------------------------------------------
# Problem 3
# ------------------------------------------------------------
# Change in pp for different alpha value
alpha_range = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
uni_perp_alpha = []
bi_perp_alpha = []
for alpha in alpha_range:
    output = get_pp(alpha, text_train, text_test)
    uni_perp_alpha.append(output[0])
    bi_perp_alpha.append(output[1])

print(uni_perp_alpha)
print(bi_perp_alpha)
    
plt.plot(range(0, len(alpha_range)), uni_perp_alpha)
plt.ylabel("Unigram Perplexity")
plt.xticks(range(0, len(alpha_range)), [str(v) for v in alpha_range])
plt.title("Change in unigram perplexity")
plt.show()
plt.plot(range(0, len(alpha_range)), bi_perp_alpha)
plt.ylabel("Bigram Perplexity")
plt.xticks(range(0, len(alpha_range)),[str(v) for v in alpha_range])
plt.title("Change in Bigram Perplexity")
plt.show()
