from __future__ import division
from data import *
from submitters_details import get_details
import tester
from collections import defaultdict
import operator
import numpy as np


def most_frequent_train(train_data):
    """
        Gets training data that includes tagged sentences.
        Returns a dictionary that maps every word in the training set to its most frequent tag.
    """
    # YOUR CODE HERE
    words_label_freq = defaultdict(lambda: defaultdict(int))
    words_most_freq = {}
    for sent in train_data:
        for token in sent:
            words_label_freq[token[0]][token[1]] += 1

    for word, labels_dict in words_label_freq.iteritems():
        words_most_freq[word] = max(labels_dict.iteritems(), key=operator.itemgetter(1))[0]

    return words_most_freq
    # END YOUR CODE


def frequent_train(train_data, tag_to_idx_dict):
    """
        Gets test data and tag prediction map.
        Returns an evaluation of the accuracy of the most frequent tagger.
    """
    # YOUR CODE HERE
    words_label_freq = {}
    num_of_tags = 46

    for sent in train_data:
        for token in sent:
            if token[0] in words_label_freq:
                words_label_freq[token[0]][tag_to_idx_dict[token[1]]] += 1
            else:
                words_label_freq[token[0]] = [0 for _ in range(num_of_tags)]
                words_label_freq[token[0]][tag_to_idx_dict[token[1]]] += 1

    for word in words_label_freq:
        sum_freq = sum(words_label_freq[word])
        words_label_freq[word] = [word_tag / sum_freq for word_tag in words_label_freq[word]]

    return words_label_freq
    # END YOUR CODE


def most_frequent_eval(test_set, pred_tags):
    error_count = 0
    total_count = 0
    for sent in test_set:
        for token in sent:
            if token[1] != pred_tags[token[0]]:
                error_count += 1
            total_count += 1
    return str((total_count - error_count) / total_count)


def softmax(x):
    x = np.asarray(x)
    x = x.astype(float)
    if x.ndim == 1:
        S = np.sum(np.exp(x))
        return np.exp(x) / S
    elif x.ndim == 2:
        result = np.zeros_like(x)
        M, N = x.shape
        for n in range(N):
            S = np.sum(np.exp(x[:, n]))
            result[:, n] = np.exp(x[:, n]) / S
        return result.tolist()


def values_to_propabillity(x):
    for vlistidx in range(len(x)):
        sum_vlist = sum(x[vlistidx])
        x[vlistidx] = [keyv / sum_vlist if sum_vlist > 0 else 0 for keyv in x[vlistidx]]
    return x


if __name__ == "__main__":
    print (get_details())
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")
    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    model = most_frequent_train(train_sents)
    print "dev: most frequent acc: " + most_frequent_eval(dev_sents, model)

    tester.verify_most_frequent_model(model)

    if os.path.exists('Penn_Treebank/test.gold.conll'):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        print "test: most frequent acc: " + most_frequent_eval(test_sents, model)
