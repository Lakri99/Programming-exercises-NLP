import string
import re
import matplotlib.pyplot as plt
import math

from collections import Counter
from sklearn.model_selection import train_test_split

def preprocessing(text):
    word_list = [word for word in text.split() if word.isalpha()]
    return word_list

lang_corpus = ["bg","de","fi","fr"]
for lang in lang_corpus:
    filename = './exercise5_corpora/corpus.'+lang
    with open(filename) as f:
        text = f.read()
    f.close()
    words = preprocessing(text)
    # Splitting test and train
    train_words ,test_words = train_test_split(words,test_size=0.2) 
    oov_list = []
    vocab_len = list(range(1000,11000, 1000))
    for vocab_size in vocab_len:
        vocab = Counter(train_words).most_common(vocab_size)
        vocab_words = [item[0] for item in vocab]
        counter = 0
        # Calculate oov rate
        for word in test_words:
            if word not in vocab_words:
                counter += 1
        oov_list.append(math.log(counter / len(test_words),2))

    # plt.figure()
    plt.plot(vocab_len, oov_list, label = lang)
    plt.ylabel("OOV rate")
    plt.xlabel("size of the vocabulary")
    plt.legend(loc="upper right")
    plt.title("OOP rate by size of vocab")
    plt.show()
    plt.savefig('oov_rate_'+lang+'.png')
