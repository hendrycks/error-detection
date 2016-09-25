# Dan Hendrycks, Feb 29th

import numpy as np
import io
import re

# Problems 1 and 2

def embeddings_to_dict(filename):
    '''
    :param filename: the file name of the word embeddings | file is assumed
    to follow this format: "word[tab]dimension 1[space]dimension 2[space]...[space]dimension 50"
    :return: a dictionary with keys that are words and values that are the embedding of a word
    '''
    with io.open(filename, 'r', encoding='utf-8') as f:
        word_vecs = {}
        for line in f:
            line = line.strip('\n').split()
            word_vecs[line[0]] = np.array([float(s) for s in line[1:]])

    return word_vecs

def data_to_mat(filename, vocab, word_ids, seq_len=36, start_tag=0, end_tag=1, pad_tag=2,
                is_not_twitter=False):
    '''
    :param filename: the filename of a training, development, devtest, or test set
    :param vocab: a list of strings, one for each embedding (the keys of a dictionary)
    :param tag_to_number: a dictionary of tags to predict and a numerical encoding of those tags;
    with this, we will predict numbers instead of strings
    :param window_size: the context window size for the left and right; thus we have 2*window_size + 1
    words considered at a time
    :param start_symbol: since the <s> symbol has no embedding given, chose a symbol in the vocab
    to replace <s>. Common choices are u'UUUNKKK' or u'</s>'
    :return: a n x (window_size*2 + 1) matrix containing context windows and the center word
    represented as strings; n is the number of examples. ALSO return a n x |tag_to_number|
    matrix of labels for the n examples with a one-hot (1-of-k) encoding
    '''
    with io.open(filename, 'r', encoding='utf-8') as f:
        x, tweet_words, y, tweet_labels = [], [], [], []
        start = True
        for line in f:

            if is_not_twitter is True:
                line = re.sub(r'[\s_]+', ' ', line).strip()
                line = line.split()
                if len(line) > 0:
                    line = line[1] + '\t' + line[2]
                else:
                    line = ''

            line = line.strip('\n')

            if len(line) == 0:              # if end of tweet
                tweet_words.extend(['**end**'])
                tweet_labels.append(end_tag)

                # ensure tweet words are in vocab; if not, map to "UUUNKKK"

                if len(tweet_words) < seq_len:
                    tweet_words += ['**pad**'] * (seq_len - len(tweet_words))
                    tweet_labels += [pad_tag] * (seq_len - len(tweet_labels))
                elif len(tweet_words) > seq_len:
                    tweet_words = tweet_words[:seq_len]
                    tweet_labels = tweet_labels[:seq_len]

                tweet_words = [word_ids[w] if w in vocab else word_ids['UUUNKKK'] for w in tweet_words]

                x.append(tweet_words)
                y.append(tweet_labels)

                tweet_words, tweet_labels = [], []
                start = True
                continue

            # if before end
            word, label = line.split('\t')

            if start:
                tweet_words.extend(['**start**'])
                tweet_labels.append(start_tag)
                start = False

            tweet_words.append(word)
            tweet_labels.append(132456795498456)  # it does not matter to us since the tag label sets are different

    return np.array(x), np.array(y)


def word_list_to_embedding(words, embeddings, embedding_dimension=50):
    '''
    :param words: an n x (2*window_size + 1) matrix from data_to_mat
    :param embeddings: an embedding dictionary where keys are strings and values
    are embeddings; the output from embeddings_to_dict
    :param embedding_dimension: the dimension of the values in embeddings; in this
    assignment, embedding_dimension=50
    :return: an n x ((2*window_size + 1)*embedding_dimension) matrix where each entry of the
    words matrix is replaced with its embedding
    '''
    m, n = words.shape
    words = words.reshape((-1))

    return np.array([embeddings[w] for w in words], dtype=np.float32).reshape(m, n*embedding_dimension)
