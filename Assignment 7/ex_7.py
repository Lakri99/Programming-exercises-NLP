import nltk
import math
# nltk.download("treebank")


from nltk.corpus import treebank
from collections import Counter
from sklearn.model_selection import train_test_split

data = treebank.tagged_words()

# def no_word_pos(dictionary):
def get_bigram(word_list):
    return [(word_list[i-1],word_list[i]) for i in range(1,len(word_list))]

def get_lambda_dot(d, N, counter_train):
    N_plus = len(counter_train)
    lambda_dot = (d / N) * N_plus
    return lambda_dot

def get_n_plus_pos_pos(train_bigram_pos, prev_pos):
    counter = 0
    for (pos_1,pos) in train_bigram_pos.keys():
        if pos_1 == prev_pos:
            counter += 1
    return counter

#Find Perplexity
def get_perplexity(word_bigram_test, p_abs_dict):
    result = 0
    s = sum(word_bigram_test.values())
    for k,v in word_bigram_test.items():
        rel_freq = v / s
        result -= rel_freq * math.log(p_abs_dict[k])  
    perplexity = math.exp(result)
    return perplexity


# Preprocessing
processed_data = [(word.lower(), tag) for (word, tag) in data if tag not in ["CD","LS", "SENT", "SYM", "#", "$", "“", "“", "-LRB-", "-RRB-", ",", ":", ".","-NONE-", "”"]]
train_data ,test_data = train_test_split(processed_data,test_size=0.2) 

# vocab = Counter(processed_data)
train_counter = Counter(train_data)
# word_counter_vocab = Counter([word for (word, tag) in processed_data])
# pos_list_vocab = [tag for (word, tag) in processed_data]
word_counter_train = Counter([word for (word, tag) in train_data])
word_bigram_test = Counter(get_bigram([word for (word, tag) in test_data]))
pos_list_train = [tag for (word, tag) in train_data]
test_counter = Counter(test_data)
tag_count_train = Counter(pos_list_train)
# tag_count_vocab = Counter(pos_list_vocab)

test_bigram_counter = Counter(get_bigram(test_data))
train_bigram_pos = Counter(get_bigram(pos_list_train))

p_abs_dict = {}
d = 0.9
N = sum(word_counter_train.values())
p_unif = 1/ len(word_counter_train)
p_unif_pos = 1/len(tag_count_train)
s = sum(word_bigram_test.values())
print(s)


for ((prev_word,prev_pos),(word,pos)) in test_bigram_counter.keys():
# Absolute discounting
    # P(W|POS)
    lambda_dot = get_lambda_dot(d, N, word_counter_train) #eq-10
    lambda_dot_pos = get_lambda_dot(d, N, tag_count_train) #eq-16
    if word not in word_counter_train:
        p_abs = lambda_dot * p_unif
        p_abs_pos = lambda_dot_pos * p_unif_pos
        p_abs_final = p_abs
        p_abs_pos_final = p_abs_pos
    else:
        p_abs = (lambda_dot * p_unif) + (max(word_counter_train.get(word,0) - d,0) / N) #eq-7
        p_abs_pos = (lambda_dot_pos * p_unif_pos) + (max(tag_count_train.get(word,0) - d,0) / N) #eq-13
        N_plus_pos = tag_count_train.get(pos) #eq-9
        N_plus_pos_pos = get_n_plus_pos_pos(train_bigram_pos, prev_pos) #eq-15
        lambda_pos = (d / tag_count_train.get(pos)) * N_plus_pos #eq-8
        lambda_pos_pos = (d / tag_count_train.get(prev_pos)) * N_plus_pos_pos #eq-14
        p_abs_final = (lambda_pos * p_abs) + (max(train_counter.get((word, pos),0)-d,0) / tag_count_train.get(pos)) #eq-6
        p_abs_pos_final = (lambda_pos_pos * p_abs_pos) + (max(train_bigram_pos.get((prev_pos, pos),0)-d,0) / tag_count_train.get(prev_pos)) #eq-12
    # print(p_abs_final, p_abs_pos_final)
    p_abs_dict[(prev_word, word)] = p_abs_final * p_abs_pos_final
# break

# Perplexity 
perplexity = get_perplexity(word_bigram_test, p_abs_dict)
print(perplexity)



    


    

    

# print(p_abs_dict)

    

