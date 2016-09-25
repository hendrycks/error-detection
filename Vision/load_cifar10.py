# code repurposed from the tf-learn library
import sys
import os
import pickle
import numpy as np
from six.moves import urllib
import tarfile

def to_categorical(y, nb_classes):
    y = np.asarray(y, dtype='int32')
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y

# load training and testing data
def load_data10(randomize=True, return_val=False, one_hot=False, dirname="cifar-10-batches-py"):
    tarpath = maybe_download("cifar-10-python.tar.gz",
                             "http://www.cs.toronto.edu/~kriz/", dirname)
    X_train = []
    Y_train = []

    for i in range(1, 6):
        fpath = os.path.join(dirname, 'data_batch_' + str(i))
        data, labels = load_batch(fpath)
        if i == 1:
            X_train = data
            Y_train = labels
        else:
            X_train = np.concatenate([X_train, data], axis=0)
            Y_train = np.concatenate([Y_train, labels], axis=0)

    X_test, Y_test = load_batch(os.path.join(dirname, 'test_batch'))

    X_train = np.dstack((X_train[:, :1024], X_train[:, 1024:2048],
                         X_train[:, 2048:])) / 255.
    X_train = np.reshape(X_train, [-1, 32, 32, 3])
    X_test = np.dstack((X_test[:, :1024], X_test[:, 1024:2048],
                        X_test[:, 2048:])) / 255.
    X_test = np.reshape(X_test, [-1, 32, 32, 3])

    if randomize is True:
        test_perm = np.array(np.random.permutation(X_test.shape[0]))
        X_test = X_test[test_perm]
        Y_test = np.asarray(Y_test)
        Y_test = Y_test[test_perm]

        perm = np.array(np.random.permutation(X_train.shape[0]))
        X_train = X_train[perm]
        Y_train = np.asarray(Y_train)
        Y_train = Y_train[perm]
    if return_val:
        X_train, X_val = np.split(X_train, [45000])     # 45000 for training, 5000 for validation
        Y_train, Y_val = np.split(Y_train, [45000])

        if one_hot:
            Y_train, Y_val, Y_test = to_categorical(Y_train, 10), to_categorical(Y_val, 10), to_categorical(Y_test, 10)
            return X_train, Y_train, X_val, Y_val, X_test, Y_test
        else:
            return X_train, Y_train, X_val, Y_val, X_test, Y_test
    else:
        if one_hot:
            Y_train, Y_test = to_categorical(Y_train, 10), to_categorical(Y_test, 10)
            return X_train, Y_train, X_test, Y_test
        else:
            return X_train, Y_train, X_test, Y_test


def load_batch(fpath):
    with open(fpath, 'rb') as f:
        d = pickle.load(f, encoding='latin1')
    data = d["data"]
    labels = d["labels"]
    return data, labels


def maybe_download(filename, source_url, work_directory):
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        print("Downloading CIFAR 10...")
        filepath, _ = urllib.request.urlretrieve(source_url + filename,
                                                 filepath)
        statinfo = os.stat(filepath)
        print(('CIFAR 10 downloaded', filename, statinfo.st_size, 'bytes.'))
        untar(filepath)
    return filepath


def untar(fname):
    if (fname.endswith("tar.gz")):
        tar = tarfile.open(fname)
        tar.extractall()
        tar.close()
        print("File Extracted in Current Directory")
    else:
        print("Not a tar.gz file: '%s '" % sys.argv[0])

if __name__ == '__main__':
    load_data10()
