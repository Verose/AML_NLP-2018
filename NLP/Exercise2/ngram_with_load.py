
#!/usr/local/bin/python

from __future__ import division
from data_utils import utils as du
import numpy as np
import pandas as pd
import csv
from collections import defaultdict
from itertools import chain
import pandas as pd
import pickle

### Choose how many top words to keep
#vocabsize = 2000

#with open('word_to_num' + '.pkl', 'rb') as f:
#      word_to_num =  pickle.load(f)

### Load the training set

S_train = np.load('test_save_s.npy')
S_dev = np.load('test_save_DEV.npy')

checkpoint = 1000
#TODO delete save gates

def count_trigram(dataset):

    with open('trigram_counts56522' + '.pkl', 'rb') as f:
      tridict =  pickle.load(f)
    trigram_counts = defaultdict(lambda: 0,tridict)

    for idx,sentence in enumerate(dataset[56522:],56522):
        for word_idx in range(2,len(sentence)):
            trigram_counts[(sentence[word_idx],sentence[word_idx-1],sentence[word_idx-2])] += 1
        if(idx % checkpoint == 0):
            print idx
            with open('trigram_counts' + str(idx) + '.pkl', 'wb') as f:
                pickle.dump(dict(trigram_counts), f, pickle.HIGHEST_PROTOCOL)
    #with open('trigram_counts' + str(len(dataset)) + '.pkl', 'wb') as f:
    #    pickle.dump(dict(trigram_counts), f, pickle.HIGHEST_PROTOCOL)
    print "end trigram_count"
    return dict(trigram_counts)

def count_bigram(dataset):
    with open('bigram_counts56522' + '.pkl', 'rb') as f:
      bigramdict =  pickle.load(f)
    bigram_counts = defaultdict(lambda: 0,bigramdict)

    for idx,sentence in enumerate(dataset[56522:],56522):
        for word_idx in range(2,len(sentence)):
            bigram_counts[(sentence[word_idx],sentence[word_idx-1])] += 1
        if(idx % checkpoint == 0):
            print idx
            with open('bigram_counts' + str(idx) + '.pkl', 'wb') as f:
                pickle.dump(dict(bigram_counts), f, pickle.HIGHEST_PROTOCOL)
    #with open('bigram_counts' + str(len(dataset)) + '.pkl', 'wb') as f:
    #    pickle.dump(dict(bigram_counts), f, pickle.HIGHEST_PROTOCOL)
    print "end bigram_count"
    return dict(bigram_counts)

def count_unigram(dataset):
    with open('unigram_counts56522' + '.pkl', 'rb') as f:
      unigramdict =  pickle.load(f)
    unigram_counts = defaultdict(lambda: 0,unigramdict)

    for idx,sentence in enumerate(dataset[56522:],56522):
        for word_idx in range(2,len(sentence)):
            unigram_counts[(sentence[word_idx])] += 1
        if(idx % checkpoint == 0):
            print idx
            with open('unigram_counts' + str(idx) + '.pkl', 'wb') as f:
                pickle.dump(dict(unigram_counts), f, pickle.HIGHEST_PROTOCOL)
    #with open('unigram_counts' + str(len(dataset)) + '.pkl', 'wb') as f:
    #    pickle.dump(dict(unigram_counts), f, pickle.HIGHEST_PROTOCOL)
    print "end unigram_count"
    return dict(unigram_counts)


def train_ngrams(dataset):
    """
        Gets an array of arrays of indexes, each one corresponds to a word.
        Returns trigram, bigram, unigram and total counts.
    """
    trigram_counts = dict()
    bigram_counts = dict()
    unigram_counts = dict()
    token_count = 0
    trigram_counts = count_trigram(dataset)
    bigram_counts = count_bigram(dataset)
    unigram_counts = count_unigram(dataset)

    #don't count * and STOP words
    token_count = len(list(chain.from_iterable(dataset))) - len(dataset) * 2

    return trigram_counts, bigram_counts, unigram_counts, token_count


def evaluate_ngrams(eval_dataset, trigram_counts, bigram_counts, unigram_counts, train_token_count, lambda1, lambda2):
    """
    Goes over an evaluation dataset and computes the perplexity for it with
    the current counts and a linear interpolation
    """
    perplexity = 0
    
    perplexity_sum = 0
    for idx,sentence in enumerate(eval_dataset):
        for word_idx in range(2,len(sentence)):
            if((sentence[word_idx],sentence[word_idx-1],sentence[word_idx-2]) in  trigram_counts):
                c_tri = trigram_counts[(sentence[word_idx],sentence[word_idx-1],sentence[word_idx-2])]
            else:
                c_tri = 0.

            if((sentence[word_idx],sentence[word_idx-1]) in  bigram_counts):
                c_bi = bigram_counts[(sentence[word_idx],sentence[word_idx-1])]
            else:
                c_bi = 0.

            if((sentence[word_idx]) in unigram_counts):
                c_uni = unigram_counts[(sentence[word_idx])]
            else:
                c_uni = 0.


            
            trivalue = lambda1 * (c_tri /c_bi) if c_bi > 0 else 0
            bivalue = lambda2 * (c_bi /c_uni) if c_uni > 0 else 0
            univalue = (1-lambda1-lambda2) * (c_uni /train_token_count)


            linear_inter_prob = trivalue + bivalue + univalue
            perplexity_sum += np.log2(linear_inter_prob)

    token_count = len(list(chain.from_iterable(eval_dataset))) - len(eval_dataset) * 2

    l = perplexity_sum / token_count
    perplexity = 2 ** -l
    return perplexity

def grid_search_lamdas(S_dev, trigram_counts, bigram_counts, unigram_counts, token_count):
    lambda_perplexity_dict = defaultdict(lambda: 0)
    for idx,lambda1 in enumerate([idx * 0.1 for idx in range(10)]):
        for idx2,lambda2 in enumerate([idx * 0.1 for idx in range(10-idx)]):
            lambda_perplexity_dict[(round(lambda1,1),round(lambda2,1),round(1-lambda1-lambda2,1))] = evaluate_ngrams(S_dev, trigram_counts, bigram_counts, unigram_counts, token_count, lambda1, lambda2)
            print str(lambda1) + "_" + str(lambda2)
            with open('grid_search_lamdas_lambda1_' + str(lambda1) + '_lambda2_' + str(lambda2) + '_lambda3_' + str(1-lambda1-lambda2) + '.pkl', 'wb') as f:
                pickle.dump(dict(lambda_perplexity_dict), f, pickle.HIGHEST_PROTOCOL)
        

    return dict(lambda_perplexity_dict)

def get_smallest_perplexity(perp_dict = None):

    perp_dict
    with open('grid_search_lamdas_lambda1_0.9_lambda2_0.0_lambda3_0.1.pkl', 'rb') as f:
      perp_dict =  pickle.load(f)

    perp_table = (pd.DataFrame(perp_dict.items(), columns=['lambda_1_2_3', 'perplexity'])).sort_values(by=['perplexity'])
    print perp_table.to_string(index=False)


def test_ngram():
    """
    Use this space to test your n-gram implementation.
    """
    #Some examples of functions usage
    trigram_counts, bigram_counts, unigram_counts, token_count = train_ngrams(S_train)
    print "#trigrams: " + str(len(trigram_counts))
    print "#bigrams: " + str(len(bigram_counts))
    print "#unigrams: " + str(len(unigram_counts))
    print "#tokens: " + str(token_count)
    #perplexity = evaluate_ngrams(S_dev, trigram_counts, bigram_counts, unigram_counts, token_count, 0.5, 0.4)
    #print "#perplexity: " + str(perplexity)
    lambda_perplexity_dict = grid_search_lamdas(S_dev, trigram_counts, bigram_counts, unigram_counts, token_count)
    print str(lambda_perplexity_dict)

    get_smallest_perplexity()


if __name__ == "__main__":
    test_ngram()
