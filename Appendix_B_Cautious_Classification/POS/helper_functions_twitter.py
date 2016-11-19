# Dan Hendrycks, Feb 29th

import numpy as np
import io

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

def data_to_mat(filename, vocab, tag_to_number, window_size=1, start_symbol=u'UUUNKKK',
                one_hot=False, return_labels=True):
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
        x, tweet_words, y = [], [], []
        start = True
        for line in f:
            line = line.strip('\n')

            if len(line) == 0:              # if end of tweet
                tweet_words.extend([u'</s>'] * window_size)

                # ensure tweet words are in vocab; if not, map to "UUUNKKK"

                #tweet_words = [w if w in vocab else w.lower() if w.lower() in vocab else u'UUUNKKK' for w in tweet_words]
                tweet_words = [w if w in vocab else u'UUUNKKK' for w in tweet_words]

                # from this tweet, add the training tasks to dataset
                # the tags were already added to y
                for i in range(window_size, len(tweet_words) - window_size):
                    x.append(tweet_words[i-window_size:i+window_size+1])

                tweet_words = []
                start = True
                continue

            # if before end
            word, label = line.split('\t')

            if start:
                tweet_words.extend([start_symbol] * window_size)
                start = False

            tweet_words.append(word)

            if return_labels is True:
                if one_hot is True:
                    label_one_hot = len(tag_to_number) * [0]
                    label_one_hot[tag_to_number[label]] += 1

                    y.append(label_one_hot)
                else:
                    y.append(tag_to_number[label])

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

# Problem 1 Part 2
def word_list_to_embedding_product(words, embeddings, embedding_dimension=50):
    '''
    :param words: an n x (2*window_size + 1) matrix from data_to_mat
    :param embeddings: an embedding dictionary where keys are strings and values
    are embeddings; the output from embeddings_to_dict
    :param embedding_dimension: the dimension of the values in embeddings; in this
    assignment, embedding_dimension=50
    :return: an n x embedding_dimension matrix where the embeddings of an example were
    the hadamard product of the embeddings occupies a row in this matrix
    '''
    m, n = words.shape
    words = words.reshape((-1))

    out = np.array([embeddings[w] for w in words], dtype=np.float32).reshape(m, n, embedding_dimension)
    return np.product(out, 1)

# Problem 1 Part 2
def word_list_to_embedding_sum(words, embeddings, embedding_dimension=50):
    '''
    :param words: an n x (2*window_size + 1) matrix from data_to_mat
    :param embeddings: an embedding dictionary where keys are strings and values
    are embeddings; the output from embeddings_to_dict
    :param embedding_dimension: the dimension of the values in embeddings; in this
    assignment, embedding_dimension=50
    :return: an n x embedding_dimension matrix where the embeddings of an example were
    elementwise summed
    '''
    m, n = words.shape
    words = words.reshape((-1))

    out = np.array([embeddings[w] for w in words], dtype=np.float32).reshape(m, n, embedding_dimension)
    return np.sum(out, 1)