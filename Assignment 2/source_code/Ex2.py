import glob, os, operator, re
import string
from math import log

from random import choices
from collections import OrderedDict, Counter
import matplotlib.pyplot as plt


#Merge all corpora into one file per language
directory_list = os.listdir("./dataset")
for folder_name in directory_list:
    read_files = glob.glob('./dataset/'+folder_name+'/*.txt')
    filename = "./dataset/"+folder_name+"_corpus.txt"
    with open(filename, "wb") as outfile:
        for f in read_files:
            with open(f, "rb") as infile:
                outfile.write(infile.read())
                infile.close()
    outfile.close()

length_corpus = {}

# Preprocessing
directory_list = ['en', 'de', 'es', 'hu', 'tr']
for dirname in directory_list:
    print("\n" + dirname)
    filename = "./dataset/"+dirname+"_corpus.txt"
    file = open(filename,'rt',encoding='utf8')
    text = file.read()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers 
    text = text.translate(str.maketrans('', '', string.digits))
#check for other special characters
    text = re.sub(r'[«–\u200b„“ˈ—]+','',text)
    # lower case
    text = text.lower()
    # tokenisation by white space
    text = text.split()


    freq_dict = Counter(text)
    
    # {language: {word:frequency}} format
    length_corpus[dirname] = freq_dict   

    # Average length of top 10 words
    top_10 = dict(freq_dict.most_common(10))
    print("Top 10 most frequent words: = " + str(top_10.keys()))
    length_top_10 = [len(word) for word in top_10.keys()]
    print("Average length of top 10 words is " + str(sum(length_top_10)/len(length_top_10)))
    
    # Average length of least occuring words
    bottom_1 = dict([(k,v) for k,v in freq_dict.items() if v == 1])
    #print(bottom_1.keys())
    length_bot_1 = [len(word) for word in bottom_1.keys()]
    print("Average length of least frequently occuring words is " + str(sum(length_bot_1)/len(length_bot_1)))
    

# Ex 1.2 - normalise and plot Rank vs Frequency
# normalised frequency = (word frequency/corpus size) * minimum corpus size
min_corpus_size = min([sum(lang.values()) for lang in length_corpus.values()])
word_counts = []
for lang in directory_list:
    # Normalize the word count
    word_counts = []
    corpus_size = sum(length_corpus[lang].values())
    for (word, count) in length_corpus[lang].items():
        normalized_count = (count / corpus_size) * min_corpus_size
        word_counts.append(normalized_count)
    
    ranking = list(range(1,len(length_corpus[lang].keys()) + 1))
    ranking = [log(rank,10) for rank in ranking]
    word_counts2 = sorted(word_counts,reverse=True)
    word_counts2 = [log(freq,10) for freq in word_counts2]
    # Plot the graph
    plt.plot(ranking, word_counts2)
    #plt.legend(lang)
    plt.xlabel('Ranking')
    # Set the y axis label of the current axis.
    plt.ylabel('Word count')
plt.legend(directory_list)    
plt.title("Frequency vs Rank (Log scale)") #comment out the logging code for normal plots    
plt.show()


#Ex 2.2

#find probability distribution of corpus
def get_prob_dist(corpus):
    counter_dict = Counter(corpus)
    result_dict = {}
    for k,v in counter_dict.items():
        result_dict[k] = v / sum(counter_dict.values())
    return result_dict

#returns random word of size length
def get_prob(length, prob_dist):
    prob_dist.pop(" ")
    population = list(prob_dist.keys())
    weights = list(prob_dist.values())
    samples = choices(population, weights, k=length)
    return samples

# Preprocessing the English corpus
file = open('./dataset/en_corpus.txt','rt',encoding='utf8')
text = file.read()
text = re.sub(r'[^a-zA-Z\s]+', '', text)
# remove \t and numbers
text = text.translate(str.maketrans('', '', string.digits))
text = re.sub(r'[\n\t\u2009]+', '', text)
text = text.lower()
en_dist = get_prob_dist(text)
print("Prob distribution of corpus")
print(en_dist)

word = get_prob(4, en_dist.copy())
print("Randomly generated word" + "".join(word))

# Probability of hello
prob_hello = en_dist['h'] * en_dist['e'] * en_dist['l'] * en_dist['l'] * en_dist['o'] * en_dist[' ']
print("\nProbability of typing 'hello' is " + str(prob_hello))


#3 Bonus Frequency Analysis

input = """PU JYFWAVNYHWOF H JHLZHY JPWOLY HSZV RUVDU HZ AOL
ZOPMA JPWOLY PZ VUL VM AOL ZPTWSLZA HUK TVZA DPKLSF RU-
VDU LUJYFWAPVU ALJOUPXBLZ. PA PZ H AFWL VM ZBIZAPABAPVU
JPWOLY PU DOPJO LHJO SLAALY PU AOL WSHPUALEA PZ YLW-
SHJLK IF H SLAALY ZVTL MPELK UBTILY VM WVZPAPVUZ KVDU
AOL HSWOHILA. MVY LEHTWSL DPAO H SLMA ZOPMA VM AOYLL K
DVBSK IL YLWSHJLK IF H HUK L DVBSK ILJVTL I. AOL TLAOVK PZ
UHTLK HMALY QBSPBZ JHLZHY DOV BZLK PA PU OPZ WYPCHAL
JVYYLZWVUKLUJL. IF NYHWOPUN AOL MYLXBLUJPLZ VM SLAA-
LYZ PU AOL JPWOLYALEA HUK IF RUVDPUN AOL LEWLJALK KPZA-
YPIBAPVU VM AOVZL SLAALYZ PU AOL VYPNPUHS SHUNBHNL VM
AOL WSHPUALEA H OBTHU JHU LHZPSF ZWVA AOL CHSBL VM AOL
ZOPMA IF SVVRPUN HA AOL KPZWSHJLTLUA VM WHYAPJBSHY ML-
HABYLZ VM AOL NYHWO. AOPZ PZ RUVDU HZ MYLXBLUJF HUHS-
FZPZ. MVY LEHTWSL PU AOL LUNSPZO SHUNBHNL AOL WSHPUALEA
MYLXBLUJPLZ VM AOL SLAALYZ L, A (BZBHSSF TVZA MYLXBLUA)
HUK X, G (AFWPJHSSF SLHZA MYLXBLUA) HYL WHYAPJBSHYSF KPZA-
PUJAPCL. DPAO AOL JHLZHY JPWOLY LUJYFWAPUN H ALEA TB-
SAPWSL APTLZ WYVCPKLZ UV HKKPAPVUHS ZLJBYPAF. AOPZ PZ
ILJHBZL ADV LUJYFWAPVUZ VM ZOPMA H HUK ZOPMA I DPSS IL
LXBPCHSLUA AV H ZPUNSL LUJYFWAPVU DPAO ZOPMA H + I. PU
THAOLTHAPJHS ALYTZ AOL ZLA VM LUJYFWAPVU VWLYHAPVUZ
BUKLY LHJO WVZZPISL RLF MVYTZ H NYVBW BUKLY JVTWVZPA-
PVU"""
input = re.sub(r'[^a-zA-Z\s]+', '', input)
input = re.sub(r'[\n\t\u2009]+', '', input)
ip_dist = get_prob_dist(input)

print("Prob distribution of input")
print(ip_dist)
en_probs = [k.upper() for k,v in sorted(ip_dist.items(), key=lambda item: item[1])]
de_probs = [k.upper() for k,v in sorted(en_dist.items(), key=lambda item: item[1])]
decoded_text = ""

for char in input:
    if char in en_probs:
        ind = en_probs.index(char)
        decoded_text = decoded_text + de_probs[ind]
    else:
        decoded_text = decoded_text + char

print("\nEncoded Text = \n" + input)
print("\nDecoded Text = \n" + decoded_text)

def swap(a,b,mystring):
    temp = '0'
    return mystring.replace(a,temp).replace(b,a).replace(temp,b)

# To get actually deciphered text, following characters have to be swapped
s2 = swap('A','I', decoded_text)
s2 = swap('I', 'T', s2)
s2 = swap('C', 'D', s2)
s2 = swap('R', 'S', s2)
s2 = swap('Y', 'F', s2)
s2 = swap('J', 'K', s2)
s2 = swap('S', 'O', s2)
s2 = swap('F', 'W', s2)
s2 = swap('D', 'P', s2)
s2 = swap('F', 'G', s2)
s2 = swap('M', 'F', s2)
s2 = swap('J', 'Q', s2)
s2 = swap('V', 'X', s2)
print("\nDecoded text = \n" + s2)