# import nltk
# nltk.download('senseval')

from nltk.corpus import senseval
from collections import Counter
import time

import numpy as np

hard_f, interest_f, line_f, serve_f = senseval.fileids()

class sample(object):
    def __init__(self, inst):
        self.label=inst.senses[0]

        p = inst.position
        context = [tuple[0]  for tuple in inst.context[p-5:p] if len(tuple)>1] #checking if list element is actually a tuple, phrasal elements are not
        context+= [tuple[0] for tuple in inst.context[p+1:p+6] if len(tuple)>1]
        self.context=context

    def context_to_index(self, word_to_id):
        self.context=context_to_id(self.context, word_to_id)


class EM(object):
    def __init__(self, V,K):
        """
        Randomly initializes the priors and the class conditional probabilitied
        :param samples: list of sample objects
        :set:    self.probs: vocabulary_size * num_senses sized matrix, i.e. each column is a class conditional probability distribution
                 self.priors: vector of prior probabilities
                
        """
        priors=np.random.rand(K)
        #normalize
        priors=np.divide(priors, np.sum(priors))

        probs=np.random.rand(V, K)
        #normalize
        probs=np.divide(probs, np.sum(probs, axis=0))
        
        



        self.priors=priors
        self.probs=probs

    def E_step(self,samples):
        """"
        TO DO
        E-step
        :param samples: list of sample objects
        :return:  H is a matrix of size sample_size * num_senses
                H[i,k] = h_{i,k} from the slides
        """

        probs=self.probs
        priors=self.priors
        H=np.random.rand(len(samples),len(priors))

        for i in range(len(samples)):
            context_index=samples[i].context
            words_given_sense=probs[context_index, :]
            context_given_sense=np.prod(words_given_sense, axis=0)
            #multiply by priors
            context_probs=np.multiply(context_given_sense, priors)
            for k in range(len(priors)):
                H[i][k] = (priors[k] * context_given_sense[k])/ sum(context_probs)
            break
        return H

    def M_step(self,H, C):
        """
        TO DO
        M step 
        Update self.priors and self.probs
        """
        priors=self.priors
        probs=self.probs

        np.matmul(C.transpose(), H, out=probs)
        probs = probs/ probs.sum()
        z_ = H.sum()
        priors = H.sum(axis=0) / z_

        self.priors=priors
        self.probs=probs

    def run(self, samples, C):
        """
        Iterates E step and M step until convergence
        param: samples: list of sample objects
               C: num_samples x vocablary_size matrix where each row is contains the word counts in a given context
        return: labels: final clustering

        """

        #initial log likelihood
        ll=log_likelihood(samples,self.probs,self.priors)

        while True:
            #E-step        
            H=self.E_step(samples)

            #M-step
            self.M_step(H,C)          

            old_ll=ll

            ll=log_likelihood(samples, self.probs, self.priors)
            print(ll)
            if abs(ll - old_ll) < 1e-5:
                break
            break


        labels=np.argmax(H, axis=1)
        return labels




def create_vocab(samples):
    words=set([w for s in samples for w in s.context])
    word_to_id=dict(zip(words, range(len(words))))

    senses=set([s.label for s in samples])

    return word_to_id, len(word_to_id), len(senses)


def context_to_id(context, word_to_id):
    return np.array([word_to_id[w] for w in context])


def log_likelihood(samples, probs, priors):
    """

    :param samples: list of sample objects
    :param probs: vocab_size x num_senses sized matrix containing class conditional distributions
    :param priors: num_senses long vector with prior probs
    :return: log likelihood of corpus
    """
    ll = 0
    for sample in samples:
        context_index=sample.context
        words_given_sense=probs[context_index, :]
        context_given_sense=np.prod(words_given_sense, axis=0)
        #multiply by priors
        context_probs=np.multiply(context_given_sense, priors)
        marginal=sum(context_probs)
        ll+=np.log(marginal)
        break

    return ll

def counts_matrix(samples, V):
    """
    :param samples: list of sample objects
    :param V: length of vocabulary
    :return: num_samples x vocablary_size matrix where each row is contains the word counts in a given context
    """
    C=np.zeros([len(samples), V])
    iter=0
    for sample in samples:
        context_index=sample.context
        freq_dict=Counter(context_index)
        tuples=[tuple([x,y]) for x, y in freq_dict.items()]
        ids, counts=zip(*tuples)

        C[iter, ids]=counts
        iter+=1

    return C



        



if __name__=="__main__":

    t0 = time.time()
    instances= senseval.instances(hard_f)
    # all training samples as a list
    samples=[sample(inst) for inst in instances]

    #V is size of Vocab, K is number of clusters
    word_to_id, V, K=create_vocab(samples)

    #convert contexts to indices so they can be used for indexing
    for sample in samples:
        sample.context_to_index(word_to_id)

    # C is a sample_size * vocab_size matrix
    C = counts_matrix(samples, V)
   


    # initialize vj|s, priors
    EM = EM(V, K)  

    #run the model

    labels=EM.run(samples,C)


   
    clusters=[[] for i in range(K)]
    for i in range(len(samples)):
        cl_i=labels[i]
        clusters[cl_i].append(samples[i])

    i = 0
    for cluster in clusters:
        sense_counts= Counter([sample.label for sample in cluster])
        ordered = sense_counts.most_common()
        print("ordered list of senses within cluster {0}: ".format(i),ordered)
        i += 1
    t1 = time.time()
    total_time = t1-t0
    print("Time: ", total_time)
















