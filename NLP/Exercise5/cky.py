from collections import defaultdict

from PCFG import PCFG
import math


def load_sents_to_parse(filename):
    sents = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line:
                sents.append(line)
    return sents


def cky(pcfg, sent):
    # YOUR CODE HERE
    pi = defaultdict(float)
    bp = {}
    rules = pcfg._rules
    words = sent.split()
    n = len(words)

    # initialization
    for i in range(n):
        w_i = words[i]
        parse_ok = False
        for X, derivations in rules.iteritems():
            for derivation in derivations:
                # check if  (X->w_i is a rule), otherwise its 0 from the defaultdict
                if [w_i] == derivation[0]:
                    pi[(i, i, X)] = derivation[1]
                    bp[(i, i, X)] = '({} {})'.format(X, w_i)
                    parse_ok = True
        if not parse_ok:
            return "FAILED TO PARSE!"

    # algorithm
    for i in reversed(range(n)):
        for l in range(n-i):
            j = i + l
            for X, derivations in rules.iteritems():
                max_q = 0
                for derivation in derivations:
                    if len(derivation[0]) < 2:
                        continue
                    # find max and update bp
                    for s in range(i, j):
                        Y = derivation[0][0]
                        Z = derivation[0][1]
                        curr_q = derivation[1] * pi[(i, s, Y)] * pi[(s+1, j, Z)]
                        max_q = max(curr_q, max_q)

                        if curr_q == max_q and curr_q > 0:
                            pi[(i, j, X)] = curr_q
                            bp[(i, j, X)] = '({} {} {})'.format(X, bp[(i, s, Y)], bp[(s+1, j, Z)])

    return bp[(0, n-1, 'ROOT')] if (0, n-1, 'ROOT') in bp else "FAILED TO PARSE!"
    # END YOUR CODE


if __name__ == '__main__':
    import sys
    pcfg = PCFG.from_file_assert_cnf(sys.argv[1])
    sents_to_parse = load_sents_to_parse(sys.argv[2])
    for sent in sents_to_parse:
        print cky(pcfg, sent)
