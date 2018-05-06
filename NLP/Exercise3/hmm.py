import math
import os
import time

import collections

from data import *
from submitters_details import get_details
from tester import verify_hmm_model


BEGIN_TAG = "START"
STOP_TAG = "STOP"


def hmm_train(sents):
    """
        sents: list of tagged sentences
        Returns: the q-counts and e-counts of the sentences' tags, total number of tokens in the sentences
    """

    print "Start training"
    total_tokens = 0
    q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts = \
        collections.defaultdict(int), collections.defaultdict(int), collections.defaultdict(int), \
        collections.defaultdict(int), collections.defaultdict(int)
    # YOUR CODE HERE
    # emission counts
    for sent in sents:
        total_tokens += len(sent) + 1

        for word, tag in sent:
            e_tag_counts[tag] += 1
            e_word_tag_counts[(word, tag)] += 1

    # q counts
    for sent in sents:
        tags = [BEGIN_TAG, BEGIN_TAG] + [tag for word, tag in sent] + [STOP_TAG]

        for w1 in tags[1:]:  # unigram counts BEGIN_TAG once, and the STOP_TAG
            q_uni_counts[(w1,)] += 1

        for w2, w1 in zip(tags[:-1], tags[1:]):  # bigram counts BEGIN_TAG twice, but not STOP_TAG
            q_bi_counts[(w2, w1)] += 1

        for w3, w2, w1 in zip(tags[:-2], tags[1:-1], tags[2:]):
            q_tri_counts[(w3, w2, w1)] += 1

    # END YOUR CODE
    return total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts


def e(word, tag):
    return float(e_word_tag_counts[(word, tag)]) / e_tag_counts[tag]


def q(tag3, tag2, tag1, lambda1, lambda2):
    lambda3 = 1 - lambda1 - lambda2
    probability = 0
    probability += lambda3 * (q_uni_counts[(tag1,)] / float(total_tokens))
    if q_uni_counts[(tag2,)] > 0:
        probability += lambda2 * (q_bi_counts[(tag2, tag1)] / float(q_uni_counts[(tag2,)]))
    if q_bi_counts[(tag3, tag2)] > 0:
        probability += lambda1 * (q_tri_counts[(tag3, tag2, tag1)] / float(q_bi_counts[(tag3, tag2)]))
    return probability


def hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts, lambda1,
                lambda2):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Returns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))
    # YOUR CODE HERE
    level = {(BEGIN_TAG, BEGIN_TAG): 0}
    bp = []
    low_val = -99999.
    tags = e_tag_counts.keys()

    for word in sent:
        next_level = collections.defaultdict(lambda: low_val)
        curr_bp = {}
        for v in tags:
            curr_e = e(word, v)
            if curr_e == 0:
                continue
            for (w, u), prev_log_p in level.iteritems():
                log_p_wuv = (prev_log_p + math.log(q(w, u, v, lambda1, lambda2)) + math.log(curr_e))

                if log_p_wuv > next_level[(u, v)]:
                    next_level[(u, v)] = log_p_wuv
                    curr_bp[(u, v)] = w
        bp.append(curr_bp)
        level = {key: value for key, value in next_level.iteritems() if value > low_val}

    last_tags = max(level.items(), key=lambda ((u, v), log_p): log_p + math.log(q(u, v, STOP_TAG, lambda1, lambda2)))[0]

    predicted_tags = list(last_tags)
    for curr_bp in bp[::-1][:-2]:
        w = curr_bp[tuple(predicted_tags[:2])]
        predicted_tags.insert(0, w)
    # END YOUR CODE
    return predicted_tags


def hmm_eval(test_data, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts,
             lambda1, lambda2):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm
    """
    print "Start evaluation"
    acc_viterbi = 0.0
    # YOUR CODE HERE
    correct = 0
    count = 0

    for sent in test_data:
        words = [word for word, tag in sent]
        hmm_tags = hmm_viterbi(words, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,
                               e_tag_counts, lambda1, lambda2)

        for (word, tag), hmm_tag in zip(sent, hmm_tags):
            if tag == hmm_tag:
                correct += 1
        count += len(sent)

    acc_viterbi = float(correct) / count
    # END YOUR CODE

    return str(acc_viterbi)


if __name__ == "__main__":
    print (get_details())
    start_time = time.time()
    train_sents = read_conll_pos_file("data/Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("data/Penn_Treebank/dev.gold.conll")
    vocab = compute_vocab_count(train_sents)

    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    lambda1 = 0.7
    lambda2 = 0.25
    total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts = hmm_train(train_sents)
    verify_hmm_model(total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts)
    acc_viterbi = hmm_eval(dev_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,
                           e_tag_counts, lambda1, lambda2)
    print "Dev: Accuracy of Viterbi hmm: " + acc_viterbi

    train_dev_time = time.time()
    print "Train and dev evaluation elapsed: " + str(train_dev_time - start_time) + " seconds"

    if os.path.exists("data/Penn_Treebank/test.gold.conll"):
        test_sents = read_conll_pos_file("data/Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        acc_viterbi = hmm_eval(test_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,
                               e_tag_counts, lambda1, lambda2)
        print "Test: Accuracy of Viterbi hmm: " + acc_viterbi
        full_flow_end = time.time()
        print "Full flow elapsed: " + str(full_flow_end - start_time) + " seconds"
