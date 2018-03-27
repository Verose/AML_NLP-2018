#!/usr/bin/env python


# TODO d=3
# delete [target,:] if didn't help

import random

import numpy as np

from q1b_softmax import softmax
from q1d_sigmoid import sigmoid
from q1e_gradcheck import gradcheck_naive


def normalizeRows(x):
    norm_vec = np.linalg.norm(x, 2, 1)
    x = x / norm_vec[:, None]

    return x


def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0, 4.0], [1, 2]]))
    print x
    ans = np.array([[0.6, 0.8], [0.4472136, 0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print ""


def softmaxCostAndGradient(predicted, target, output_vectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)
    target -- integer, the index of the target word
    outputVectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    grad_pred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """

    # x_i(hat) for the sigmoid is u_o^Tv_i so x(hat) is vu^T
    x_hat = np.matmul(predicted, output_vectors.T)
    y_hat = softmax(x_hat)
    # target index refers to y hot vecor
    cost = -(np.log(y_hat[target]))

    # using the derivative of the cost from 2a
    grad_pred = np.matmul(y_hat, output_vectors) - output_vectors[target]
    # using the derivative of the cost from 2b
    grad = np.outer(predicted, y_hat).transpose()
    grad[target] -= predicted

    return cost, grad_pred, grad


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in xrange(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))

    y_hat = sigmoid(np.matmul(outputVectors, predicted))

    # We use here that sigmoid(-x) = 1-sigmoid(x) as proved in the PDF
    cost = -np.log(y_hat[target]) - np.sum([np.log(1 - y_hat[indices[1:]])])

    grad_pred = -(1 - y_hat[target]) * outputVectors[target]
    grad_pred += np.sum((y_hat[indices[1:]] * outputVectors[indices[1:]].transpose()).transpose(), 0)

    # for all non negative sampling or target the grad is zero
    grad = np.zeros(shape=outputVectors.shape)
    # for target
    grad[target, :] = (-1) * (1 - y_hat[target]) * predicted
    # for negative samples
    for negative_idx in range(1, K + 1):
        grad[indices[negative_idx]] += y_hat[indices[negative_idx]] * predicted

    return cost, grad_pred, grad


def skipgram(current_word, C, context_words, tokens, input_vectors, output_vectors,
             dataset, word2vec_cost_and_gradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    cost = 0.0
    grad_in = np.zeros(input_vectors.shape)
    grad_out = np.zeros(output_vectors.shape)

    for context_idx, context_word in enumerate(context_words):
        cur_cost, cur_grad_pred, cur_grad = word2vec_cost_and_gradient(input_vectors[tokens[current_word]],
                                                                       tokens[context_word], output_vectors, dataset)
        cost += cur_cost
        grad_in[tokens[current_word]] += cur_grad_pred
        grad_out += cur_grad

    return cost, grad_in, grad_out


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vec_model, tokens, word_vectors, dataset, C,
                         word2vec_cost_and_gradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(word_vectors.shape)
    N = word_vectors.shape[0]
    input_vectors = word_vectors[:N / 2, :]
    output_vectors = word_vectors[N / 2:, :]
    for i in xrange(batchsize):
        C1 = random.randint(1, C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vec_model == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vec_model(
            centerword, C1, context, tokens, input_vectors, output_vectors,
            dataset, word2vec_cost_and_gradient)
        cost += c / batchsize / denom
        grad[:N / 2, :] += gin / batchsize / denom
        grad[N / 2:, :] += gout / batchsize / denom

    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()

    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0, 4)], [tokens[random.randint(0, 4)] for _ in xrange(2 * C)]

    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10, 3))
    dummy_tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])
    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
                    dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
                    dummy_vectors)

    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
                   dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset)
    print skipgram("c", 1, ["a", "b"],
                   dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset,
                   negSamplingCostAndGradient)


if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()
