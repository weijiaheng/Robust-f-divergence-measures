import numpy as np
import torch
import os
import codecs
#import torch.nn as nn
from sklearn.metrics import confusion_matrix
import keras
from numpy.testing import assert_array_almost_equal



def unbiased_edge(x, y, p_minus, p_plus):
    z = (y - (p_minus - p_plus)) * x
    return z / (1 - p_minus - p_plus)


def unbiased_mean_op(X, y, p_minus, p_plus):
    return np.array([unbiased_edge(X[i, :], y[i], p_minus, p_plus)
                    for i in np.arange(X.shape[0])]).mean(axis=0)


def build_uniform_P(size, noise):
    """ The noise matrix flips any class to any other with probability
    noise / (#class - 1).
    """

    assert(noise >= 0.) and (noise <= 1.)

    P = noise / (size - 1) * np.ones((size, size))
    np.fill_diagonal(P, (1 - noise) * np.ones(size))

    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P


def build_for_cifar100(size, noise):
    """ The noise matrix flips to the "next" class with probability 'noise'.
    """

    assert(noise >= 0.) and (noise <= 1.)

    P = (1. - noise) * np.eye(size)
    for i in np.arange(size - 1):
        P[i, i+1] = noise

    # adjust last row
    P[size-1, 0] = noise

    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P


def row_normalize_P(P, copy=True):

    if copy:
        P_norm = P.copy()
    else:
        P_norm = P

    D = np.sum(P, axis=1)
    for i in np.arange(P_norm.shape[0]):
        P_norm[i, :] /= D[i]
    return P_norm


def noisify(y, p_minus, p_plus=None, random_state=0):
    """ Flip labels with probability p_minus.
    If p_plus is given too, the function flips with asymmetric probability.
    """

    assert np.all(np.abs(y) == 1)

    m = y.shape[0]
    new_y = y.copy()
    coin = np.random.RandomState(random_state)

    if p_plus is None:
        p_plus = p_minus

    # This can be made much faster by tossing all the coins and completely
    # avoiding the loop. Although, it is not simple to write the asymmetric
    # case then.
    for idx in np.arange(m):
        if y[idx] == -1:
            if coin.binomial(n=1, p=p_minus, size=1) == 1:
                new_y[idx] = -new_y[idx]
        else:
            if coin.binomial(n=1, p=p_plus, size=1) == 1:
                new_y[idx] = -new_y[idx]

    return new_y


def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """

    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y



def build_matrix_pair(num_classes, n_pair, noise_pair):
    '''
    (2,3) (0.2, 0.6)
    M_23 = 0.2 M_32 = 0.6
    '''

    #estimate noise matrix with clean
    nm = np.array(
        [[0.7, 0.3, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,],
         [0.4, 0.6, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,],
         [0. , 0. , 0.3, 0.7, 0. , 0. , 0. , 0. , 0. , 0. ,],
         [0. , 0. , 0.1, 0.9, 0. , 0. , 0. , 0. , 0. , 0. ,],
         [0. , 0. , 0. , 0. , 0.9, 0. , 0. , 0. , 0.1, 0. ,],
         [0. , 0. , 0. , 0. , 0. , 0.8, 0. , 0.2, 0. , 0. ,],
         [0. , 0. , 0. , 0. , 0. , 0. , 0.5, 0. , 0. , 0.5,],
         [0. , 0. , 0. , 0. , 0. , 0.6, 0. , 0.4, 0. , 0. ,],
         [0. , 0. , 0. , 0. , 0.8, 0. , 0. , 0. , 0.2, 0. ,],
         [0. , 0. , 0. , 0. , 0. , 0. , 0.3, 0. , 0. , 0.7,]]
    )
    noise_matrix = nm.transpose()

    return noise_matrix


def build_matrix(nb_classes, noise_list):
    noise_list_sum = np.sum(noise_list)
    s = (nb_classes,  nb_classes)
    noise_matrix = np.zeros(s)
    for i in range(nb_classes):
        for j in range(nb_classes):
            if i != j:
                noise_matrix[i][j] = noise_list[i]
            else:
                noise_matrix[i][j] = 1. - noise_list_sum + noise_list[i]

    return noise_matrix


def flipped_index(traget, original_label, num_class, noise_matrix, random_state):
    num_class = len(np.unique(original_label))
    # Partial ratio p: flip p/1 of orginal labels to noisy labels
    label = original_label.copy()
    flip_index = []
    num = np.count_nonzero(label == traget)
    flipper = np.random.RandomState(random_state)
    for j in range(num_class):
        tmp_idx = np.where(label == traget)[0]
        flip_i = flipper.choice(tmp_idx, int(num * noise_matrix[j][traget]), replace=False)
        label[flip_i] = 10086
        flip_index.append(flip_i)
    return flip_index

def add_noise(original_label, num_class, noise_matrix, random_state=10086):
    original_label = np.asarray(original_label)
    label = original_label.copy()
    noisy_label = original_label.copy()
    for i in range(num_class):
        flip_index_new = flipped_index(i, label, num_class, noise_matrix, random_state)
        # print(len(flip_index_new))
        for j in range(len(flip_index_new)):
            noisy_label[flip_index_new[j]] = j

    print(np.bincount(noisy_label))
    return noisy_label



def noisify_with_P(y_train, nb_classes, noise_list=None, n_pair=None, noise_pair=None, random_state=None):
    if noise_list is not None:
        P = build_matrix(nb_classes, noise_list)
    elif (n_pair is not None) and (noise_pair is not None):
        P = build_matrix_pair(nb_classes, n_pair, noise_pair)
    # seed the random numbers with #run
    y_train_noisy = add_noise(y_train, nb_classes, noise_matrix=P, random_state=random_state)
    actual_noise = (y_train_noisy != y_train).mean()
    assert actual_noise > 0.0
    print('Actual noise %.2f' % actual_noise)
    y_train = y_train_noisy

    return y_train, P.transpose()

def noisify_with_P_1(y_train, nb_classes, r, noise_list=None, n_pair=None, noise_pair=None, random_state=None):
    noise_file = f"CIFAR10_noise_r0_{int(r*10)}.pt"
    key = 'noise_label_train'
    (x_train_tmp, y_train_tmp), (x_test_tmp, y_test_tmp) = keras.datasets.cifar10.load_data()
    clean_label = y_train_tmp
    dirty_label = torch.load(os.path.join('./noise_label', noise_file))
    P = confusion_matrix(clean_label, dirty_label[key])/5000

#    P = P.transpose()
    # seed the random numbers with #run
    y_train_noisy = multiclass_noisify(y_train, P=P,
                                       random_state=random_state)
    actual_noise = (y_train_noisy != y_train).mean()
    assert actual_noise > 0.0
    print('Actual noise %.2f' % actual_noise)
    y_train = y_train_noisy

    return y_train, P

def noisify_mnist_asymmetric(y_train, noise, random_state=None):
    """mistakes:
        1 <- 7
        2 -> 7
        3 -> 8
        5 <-> 6
    """
    nb_classes = 10
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 1 <- 7
        P[7, 7], P[7, 1] = 1. - n, n

        # 2 -> 7
        P[2, 2], P[2, 7] = 1. - n, n

        # 5 <-> 6
        P[5, 5], P[5, 6] = 1. - n, n
        P[6, 6], P[6, 5] = 1. - n, n

        # 3 -> 8
        P[3, 3], P[3, 8] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy

    return y_train, P


def noisify_cifar10_asymmetric(y_train, noise, random_state=None):
    """mistakes:
        automobile <- truck
        bird -> airplane
        cat <-> dog
        deer -> horse
    """
    nb_classes = 10
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # automobile <- truck
        P[9, 9], P[9, 1] = 1. - n, n

        # bird -> airplane
        P[2, 2], P[2, 0] = 1. - n, n

        # cat <-> dog
        P[3, 3], P[3, 5] = 1. - n, n
        P[5, 5], P[5, 3] = 1. - n, n

        # automobile -> truck
        P[4, 4], P[4, 7] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy

    return y_train, P


def noisify_cifar100_asymmetric(y_train, noise, random_state=None):
    """mistakes are inside the same superclass of 10 classes, e.g. 'fish'
    """
    nb_classes = 100
    P = np.eye(nb_classes)
    n = noise
    nb_superclasses = 20
    nb_subclasses = 5

    if n > 0.0:
        for i in np.arange(nb_superclasses):
            init, end = i * nb_subclasses, (i+1) * nb_subclasses
            P[init:end, init:end] = build_for_cifar100(nb_subclasses, n)

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy

    return y_train, P


def noisify_binary_asymmetric(y_train, noise, random_state=None):
    """mistakes:
        1 -> 0: n
        0 -> 1: .05
    """
    P = np.eye(2)
    n = noise

    assert 0.0 <= n < 0.5

    if noise > 0.0:
        P[1, 1], P[1, 0] = 1.0 - n, n
        P[0, 0], P[0, 1] = 0.95, 0.05

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy

    return y_train, P
