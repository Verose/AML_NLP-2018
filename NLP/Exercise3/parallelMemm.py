from __future__ import division
from data import *
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
import time
from submitters_details import get_details
import pickle
import cPickle as pkl
import scipy.sparse
import operator
from collections import defaultdict
import code
import multiprocessing
import numpy
from most_frequent import *


def extract_features_base(curr_word, next_word, next_tag, prev_word, prevprev_word, prev_tag, prevprev_tag):
    """
        Receives: a word's local information
        Returns: The word's features.
    """
    features = {}
    features['word'] = curr_word
    features['wordlen'] = len(curr_word)
    features['prevwordlen'] = len(prev_word)
    prefix_suffix_len = 5 if len(curr_word)  > 5 else len(curr_word)
    for idx in range(1, prefix_suffix_len + 1):
        features['suffix' + str(idx)] = curr_word[-idx:]
        features['prefix' + str(idx)] = curr_word[:idx]

    features['trigram'] = prev_tag + prevprev_tag
    features['bigram'] = prev_tag
    features['uigram'] = ''

    features['prevword'] = prev_word
    features['nextword'] = next_word
    features['prevprevword'] = prevprev_word
    features['prevword_prevtag'] = prev_word+prev_tag
    features['nextword_nexttag'] = next_word+next_tag

    features['capitalletter'] = (len(curr_word) > 0) and (curr_word[0].isupper())
    features['isalpha'] = curr_word.isalpha()

    return features

def extract_features(sentence, i, t = None, u = None):
    curr_word, next_token[0], next_token[1], prev_token[0], prevprev_token[0], prev_token[1], prevprev_token[1] = get_word_tag_params(sentence, i, t = None, u = None)
    
    return extract_features_base(curr_word, next_token[0], next_token[1], prev_token[0], prevprev_token[0], prev_token[1], prevprev_token[1])

def get_word_tag_params(sentence, i, t = None, u = None):
    if(t == None):
        prevprev_token = sentence[i - 2] if i > 1 else ('<s>', '*')
    else:
        prevprev_word = sentence[i - 2][0] if i > 1 else '<s>'
        prevprev_token = (prevprev_word, t)
    if(u == None):
        prev_token = sentence[i - 1] if i > 0 else ('<s>', '*')
    else:
        prev_word = sentence[i - 1][0] if i > 0 else '<s>'
        prev_token = (prev_word, u)
        
    curr_word = sentence[i][0]
    next_token = sentence[i + 1] if i < (len(sentence) - 1) else ('</s>', 'STOP')

    return curr_word, next_token[0], next_token[1], prev_token[0], prevprev_token[0], prev_token[1], prevprev_token[1]


def vectorize_features(vec, features):
    """
        Receives: feature dictionary
        Returns: feature vector

        Note: use this function only if you chose to use the sklearn solver!
        This function prepares the feature vector for the sklearn solver,
        use it for tags prediction.
    """
    example = [features]
    return vec.transform(example)

def create_examples(sents, tag_to_idx_dict):
    examples = []
    labels = []
    num_of_sents = 0
    for sent in sents:
        num_of_sents += 1
        for i in xrange(len(sent)):
            features = extract_features(sent, i)
            examples.append(features)
            labels.append(tag_to_idx_dict[sent[i][1]])

    return examples, labels

def memm_greeedy(sent, test_data_vectorized, words_count, logreg, vec, index_to_tag_dict):
    """
        Receives: a sentence to tag and the parameters learned by memm
        Returns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))
    for i, token in enumerate(sent):
        predicted_tags[i] = logreg.predict(test_data_vectorized[i+words_count])[0]
    
    return predicted_tags

def memm_viterbi(sent, test_data_vectorized, words_count, logreg, vec, index_to_tag_dict, return_dict = None, freq_word_tag = None,process_idx = None):
    """
        Receives: a sentence to tag and the parameters learned by memm
        Returns: predicted tags for the sentence
    """
    epsilon = 10**-8

    predicted_tags = [""] * (len(sent))
    DP = [numpy.zeros((len(index_to_tag_dict),len(index_to_tag_dict))) for idx in range(len(sent) + 1)]
    back_pointers = [numpy.zeros((len(index_to_tag_dict),len(index_to_tag_dict)), dtype=int) for idx in range(len(sent) + 1)]
    DP[0][len(index_to_tag_dict)-1][len(index_to_tag_dict)-1] = 1

    #first in key tuple is prev tag (u)
    for idx in range(len(sent)):

        #t
        #wordt_freq_list = freq_word_tag[sent[idx-2][0]] if idx > 1 else [0 if tagidx != (len(index_to_tag_dict)-1) else 1 for tagidx in range(len(index_to_tag_dict))]
        for keyt, valuet in index_to_tag_dict.iteritems(): 
            if(DP[idx][keyt].any() is False):
                continue    
            
            #u
            #wordu_freq_list = freq_word_tag[sent[idx-1][0]] if idx > 0 else [0 if tagidx != (len(index_to_tag_dict)-1) else 1 for tagidx in range(len(index_to_tag_dict))]
            for keyu, valueu in index_to_tag_dict.iteritems(): 
                if(DP[idx][keyt][keyu] == 0):
                    continue 
                #if(wordu_freq_list[keyu] == 0):
                #    continue
                word_tag_params = get_word_tag_params(sent,idx,keyt,keyu)
                pred_with_cur_t = logreg.predict_proba(vectorize_features(vec,extract_features_base(*word_tag_params)))[0]
                
                #v
                #wordv_freq_list = freq_word_tag[sent[idx][0]]
                for keyv, valuev in index_to_tag_dict.iteritems():
                    if(keyv == len(index_to_tag_dict) - 1):
                        continue
                    #if(wordv_freq_list[keyv] == 0):
                    #    continue

                    max_prev = DP[idx+1][keyu][keyv]
                    best_t = back_pointers[idx+1][keyu][keyv]
               
                    #if(wordt_freq_list[keyt] == 0):
                    #    continue


                    prev_DP_t_u_v_val = pred_with_cur_t[keyv] * DP[idx][keyt][keyu]
                    if(max_prev < prev_DP_t_u_v_val):
                        max_prev = prev_DP_t_u_v_val
                        best_t = keyt                
                        
                    DP[idx+1][keyu][keyv] = max_prev if max_prev > epsilon else 0
                    back_pointers[idx+1][keyu][keyv] = best_t


    max_val_idx = numpy.where(DP[len(sent)] == numpy.max(DP[len(sent)]))
    predicted_tags[len(sent) - 2], predicted_tags[len(sent) - 1] = max_val_idx[0][0],max_val_idx[1][0]
    for idx in range(len(sent) - 3,-1,-1):
        predicted_tags[idx] = back_pointers[idx+3][predicted_tags[idx+1]][predicted_tags[idx+2]]

    if(process_idx == None):
        return predicted_tags

    return_dict[process_idx] = predicted_tags

def should_add_eval_log(sentene_index):
    if sentene_index > 0 and sentene_index % 10 == 0:
        if sentene_index < 150 or sentene_index % 200 == 0:
            return True

    return False


def memm_eval(test_data, test_data_vectorized, logreg, vec, index_to_tag_dict, freq_word_tag):
    """
    Receives: test data set and the parameters learned by memm
    Returns an evaluation of the accuracy of Viterbi & greedy memm
    """

    manager = multiprocessing.Manager()
    return_dict = manager.dict()


    acc_viterbi, acc_greedy = 0.0, 0.0
    eval_start_timer = eval_mid_timer = time.time()
    total_count, greedy_wrong_count,viterbi_wrong_count = 0,0,0
    greedy_tags = []
    viterbi_tags = []

    #try to run without tag optimizations now that i know the algo works
    #try to identify where viterbi is wrong and greedy not


    for i in range(0,len(test_data),1):
        print str([token[0] for token in test_data[i]])

        memm_viterbi(test_data[i], test_data_vectorized, total_count, logreg,vec,index_to_tag_dict,return_dict,freq_word_tag, 0)
        viterbi_tags += return_dict[0]

        greedy_tags += memm_greeedy(test_data[i], test_data_vectorized, total_count, logreg,vec,index_to_tag_dict)

        return_dict = manager.dict()


        for idx, token in enumerate(test_data[i]):

            if(token[1] != index_to_tag_dict[greedy_tags[total_count + idx]]):
                greedy_wrong_count +=1
                print "greedy is wrong on " + str(token[0]) + "  " + str(token[1])


            if(token[1] != index_to_tag_dict[viterbi_tags[total_count + idx]]):
                print "viterbi is wrong on " + str(token[0]) + "  " + str(token[1])
                viterbi_wrong_count +=1

        total_count += len(test_data[i])
        acc_greedy = (total_count - greedy_wrong_count) / total_count
        acc_viterbi = (total_count - viterbi_wrong_count) / total_count

        if should_add_eval_log(i):
            if acc_greedy == 0 and acc_viterbi == 0:
                raise NotImplementedError
            eval_end_timer = time.time()
            print str.format("Sentence index: {} greedy_acc: {}    Viterbi_acc:{} , elapsed: {} ,total time(minutes): {}", str(i), str(acc_greedy), str(acc_viterbi) , str (eval_end_timer - eval_mid_timer), str ((eval_end_timer - eval_start_timer) / 60))
            pkl.dump( greedy_tags, open( "greedy_tags" + str(i) + ".p", "wb" ) )
            pkl.dump( viterbi_tags, open( "viterbi_tags" + str(i) + ".p", "wb" ) )
            eval_mid_timer = time.time()

    return str(acc_viterbi), str(acc_greedy)

def build_tag_to_idx_dict(train_sentences):
    curr_tag_index = 0
    tag_to_idx_dict = {}
    for train_sent in train_sentences:
        for token in train_sent:
            tag = token[1]
            if tag not in tag_to_idx_dict:
                tag_to_idx_dict[tag] = curr_tag_index
                curr_tag_index += 1

    tag_to_idx_dict['*'] = curr_tag_index
    return tag_to_idx_dict

num_train_examples = 950028
num_dev_examples = 40117

if __name__ == "__main__":
    full_flow_start = time.time()
    print (get_details())
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")

    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    pkl.dump( train_sents, open( "train_sents.p", "wb" ) )
    dev_sents = preprocess_sent(vocab, dev_sents)
    pkl.dump( dev_sents, open( "dev_sents.p", "wb" ) )

    tag_to_idx_dict = build_tag_to_idx_dict(train_sents)
    pkl.dump( tag_to_idx_dict, open( "tag_to_idx_dict.p", "wb" ) )
    index_to_tag_dict = invert_dict(tag_to_idx_dict)
    pkl.dump( index_to_tag_dict, open( "index_to_tag_dict.p", "wb" ) )
    freq_word_tag = frequent_train(train_sents, tag_to_idx_dict)
    pkl.dump( freq_word_tag, open( "freq_word_tag.p", "wb" ) )
    

        # The log-linear model training.
        # NOTE: this part of the code is just a suggestion! You can change it as you wish!

    vec = DictVectorizer()
    print "Create train examples"
    train_examples, train_labels = create_examples(train_sents, tag_to_idx_dict)
    num_train_examples = len(train_examples)
    print "#example: " + str(num_train_examples)
    print "Done"

    print "Create dev examples"
    dev_examples, dev_labels = create_examples(dev_sents, tag_to_idx_dict)
    num_dev_examples = len(dev_examples)
    print "#example: " + str(num_dev_examples)
    print "Done"

    all_examples = train_examples
    all_examples.extend(dev_examples)

    pkl.dump( all_examples, open( "all_examples.p", "wb" ) )

    all_examples = pkl.load( open ("all_examples.p", "rb") )
    index_to_tag_dict = pkl.load( open ("index_to_tag_dict.p", "rb") )
    tag_to_idx_dict = pkl.load( open ("tag_to_idx_dict.p", "rb") )
    train_sents = pkl.load( open ("train_sents.p", "rb") )
    dev_sents = pkl.load( open ("dev_sents.p", "rb") )
    freq_word_tag = pkl.load( open ("freq_word_tag.p", "rb") )


    print "Vectorize examples"
    all_examples_vectorized = vec.fit_transform(all_examples)
    with open('vec.bin', 'wb') as f:
       pkl.dump(vec, f)
    scipy.sparse.save_npz('all_examples_vectorized.npz', all_examples_vectorized)

    train_examples_vectorized = all_examples_vectorized[:num_train_examples]
    scipy.sparse.save_npz('train_examples_vectorized.npz', train_examples_vectorized)

    dev_examples_vectorized = all_examples_vectorized[num_train_examples:]
    scipy.sparse.save_npz('dev_examples_vectorized.npz', dev_examples_vectorized)

    print "Done"

    logreg = linear_model.LogisticRegression(
        multi_class='multinomial', max_iter=128, solver='lbfgs', C=100000, verbose=1)
    print "Fitting..."
    start = time.time()
    logreg.fit(train_examples_vectorized, train_labels)
    end = time.time()
    print "End training, elapsed " + str(end - start) + " seconds"
    filename = 'memm_logreg.sav'
    pickle.dump(logreg, open(filename, 'wb'))
        # End of log linear model training

        # Evaluation code - do not make any changes
    logreg = pickle.load(open("memm_logreg.sav", 'rb'))
    all_examples_vectorized = scipy.sparse.load_npz('all_examples_vectorized.npz')
    train_examples_vectorized = scipy.sparse.load_npz('train_examples_vectorized.npz')
    dev_examples_vectorized = scipy.sparse.load_npz('dev_examples_vectorized.npz')
    with open('vec.bin', 'rb') as f:
       vec =  pkl.load(f)
    start = time.time()
    print "Start evaluation on dev set"
    
    acc_viterbi, acc_greedy = memm_eval(dev_sents, dev_examples_vectorized, logreg, vec, index_to_tag_dict,freq_word_tag)
    end = time.time()
    print "Dev: Accuracy greedy memm : " + acc_greedy
    print "Dev: Accuracy Viterbi memm : " + acc_viterbi

    print "Evaluation on dev set elapsed: " + str(end - start) + " seconds"
    if os.path.exists('Penn_Treebank/test.gold.conll'):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        start = time.time()
        print "Start evaluation on test set"
        acc_viterbi, acc_greedy = memm_eval(test_sents, logreg, vec, index_to_tag_dict,train_set)
        end = time.time()

        print "Test: Accuracy greedy memm: " + acc_greedy
        print "Test:  Accuracy Viterbi memm: " + acc_viterbi

        print "Evaluation on test set elapsed: " + str(end - start) + " seconds"
        full_flow_end = time.time()
        print "The execution of the full flow elapsed: " + str(full_flow_end - full_flow_start) + " seconds"