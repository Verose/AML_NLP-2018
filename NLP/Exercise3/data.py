MIN_FREQ = 3


def invert_dict(d):
    res = {}
    for k, v in d.iteritems():
        res[v] = k
    return res


def read_conll_pos_file(path):
    """
        Takes a path to a file and returns a list of word/tag pairs
    """
    sents = []
    with open(path, "r") as f:
        curr = []
        for line in f:
            line = line.strip()
            if line == "":
                sents.append(curr)
                curr = []
            else:
                tokens = line.strip().split("\t")
                curr.append((tokens[1], tokens[3]))
    return sents


def increment_count(count_dict, key):
    """
        Puts the key in the dictionary if does not exist or adds one if it does.
        Args:
            count_dict: a dictionary mapping a string to an integer
            key: a string
    """
    if key in count_dict:
        count_dict[key] += 1
    else:
        count_dict[key] = 1


def compute_vocab_count(sents):
    """
        Takes a corpus and computes all words and the number of times they appear
    """
    vocab = {}
    for sent in sents:
        for token in sent:
            increment_count(vocab, token[0])
    return vocab


""""
    Taken from: http://people.csail.mit.edu/mcollins/6864/slides/bikel.pdf
    
    Replacing words according to the following table:
    
    Word Feature            |    Example Text           |    Intuition
    twoDigitNum             |    90                     |    Two-digit year
    fourDigitNum            |    1990                   |    Four digit year
    containsDigitAndAlpha   |    A8956-67               |    Product code
    containsDigitAndDash    |    09-96                  |    Date
    containsDigitAndSlash   |    11/9/89                |    Date
    containsDigitAndComma   |    23,000.00              |    Monetary amount
    containsDigitAndPeriod  |    1.00                   |    Monetary amount, percentage
    otherNum                |    456789                 |    Other number
    allCaps                 |    BBN                    |    Organization
    capPeriod               |    M.                     |    Person name initial
    firstWord               |    first word of sentence |    No useful capitalization information
    initCap                 |    Sally                  |    Capitalized word
    lowerCase               |    can                    |    Uncapitalized word
    other                   |    ,                      |    Punctuation marks, all other words
"""


def replace_word(word, vocab, is_first):
    """
        Replaces rare words with categories (numbers, dates, etc...)
    """
    # YOUR CODE HERE
    if word.isdigit():
        if len(word) == 2:
            return 'twoDigitNum'
        elif len(word) == 4:
            return 'fourDigitNum'
    elif any(char.isdigit() for char in word):
        if any(char.isalpha() for char in word):
            return 'containsDigitAndAlpha'
        elif '-' in word:
            return 'containsDigitAndDash'
        elif '/' in word or '\\' in word:
            return 'containsDigitAndSlash'
        elif '.' in word:
            return 'containsDigitAndPeriod'
    if word.isdigit():
        return 'othernum'
    elif word.isupper():
        if '.' in word:
            return 'capPeriod'
        return 'allCaps'
    elif is_first:
        if vocab.get(word.lower(), 0) >= MIN_FREQ:  # check if word in lowercase is frequent
            return word.lower()
        else:
            return 'firstWord'
    elif word.istitle():
        return 'initCap'
    elif word.islower():
        return 'lowercase'
    # END YOUR CODE
    return "UNK"


def preprocess_sent(vocab, sents):
    """
        return a sentence, where every word that is not frequent enough is replaced
    """
    res = []
    total, replaced = 0, 0
    for sent in sents:
        new_sent = []
        is_first = True
        for token in sent:
            if token[0] in vocab and vocab[token[0]] >= MIN_FREQ:
                new_sent.append(token)
            else:
                word = replace_word(token[0], vocab, is_first)
                new_sent.append((word, token[1]))
                replaced += 1
            total += 1
            is_first = False
        res.append(new_sent)
    print "replaced: " + str(float(replaced) / total)
    return res
