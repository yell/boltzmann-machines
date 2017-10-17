#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train 2-layer Gaussian-Bernoulli-Multinomial DBM with pre-training
on CIFAR-10, augmented (x10) using shifts by 1 pixel
in all directions and horizontal mirroring.
Gaussian RBM is initialized from 26 small RBMs trained on patches 8x8
of images, as in [1].

References
----------
[1] A. Krizhevsky and G. Hinton. Learning multiple layers of features
    from tine images. 2009.
"""
print __doc__


import os
import argparse
import numpy as np
if not 'DISPLAY' in os.environ:
    import matplotlib
    matplotlib.use('Agg')

import env
from hdm.dbm import DBM
from hdm.rbm import GaussianRBM, MultinomialRBM
from hdm.utils import RNG, Stopwatch
from hdm.utils.augmentation import shift, horizontal_mirror
from hdm.utils.dataset import (load_cifar10,
                               flatten_cifar10, unflatten_cifar10)


def main():
    # training settings
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu', type=str, default='0', metavar='ID',
                        help="ID of the GPU to train on (or '' to train on CPU)")
    parser.add_argument('--n-train', type=int, default=49000, metavar='N',
                        help='number of training examples')
    parser.add_argument('--n-val', type=int, default=1000, metavar='N',
                        help='number of validation examples')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # prepare data (load + normalize + split)
    print "\nPreparing data ..."
    X, _ = load_cifar10(mode='train', path='../data/')
    X = X.astype(np.float32)
    X /= 255.
    RNG(seed=42).shuffle(X)
    n_train = min(len(X), args.n_train)
    n_val = min(len(X), args.n_val)
    X_train = X[:n_train]
    X_val = X[-n_val:]

    # augment data
    aug_data_path = '../data/X_aug.npy'
    X_aug = None

    augment = True
    if os.path.isfile(aug_data_path):
        print "\nLoading augmented data ..."
        X_aug = np.load(aug_data_path)
        print "Checking augmented data ..."
        if 10 * n_train == len(X_aug):
            augment = False

    if augment:
        print "\nAugmenting data ..."
        s = Stopwatch(verbose=True).start()

        X_aug = np.zeros((10 * n_train, 32, 32, 3), dtype=np.float32)
        X_train = unflatten_cifar10(X_train)
        X_aug[:n_train] = X_train
        for i in xrange(n_train):
            for k, offset in enumerate((
                ( 1,  0),
                (-1,  0),
                ( 0,  1),
                ( 0, -1)
            )):
                img = X_train[i].copy()
                X_aug[(k + 1) * n_train + i] = shift(img, offset=offset)
        for i in xrange(5 * n_train):
            X_aug[5 * n_train + i] = horizontal_mirror(X_aug[i].copy())

        # shuffle once again
        RNG(seed=1337).shuffle(X_aug)

        # convert to 'uint8' type to save disk space
        X_aug *= 255.
        X_aug = X_aug.astype('uint8')

        # flatten to (10 * `n_train`, 3072) shape
        X_aug = flatten_cifar10(X_aug)

        # save to disk
        np.save(aug_data_path, X_aug)

        s.elapsed()
        print "\n"

    # normalize augmented data again
    X_train = X_aug.astype(np.float32)
    X_train /= 255.
    print "Augmented shape: {0}".format(X_train.shape)
    print "Augmented range: {0}\n\n".format((X_train.min(), X_train.max()))


if __name__ == '__main__':
    main()
