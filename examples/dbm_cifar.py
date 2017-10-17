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
                               im_flatten, im_unflatten)


def main():
    # training settings
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # general
    parser.add_argument('--gpu', type=str, default='0', metavar='ID',
                        help="ID of the GPU to train on (or '' to train on CPU)")

    # data
    parser.add_argument('--n-train', type=int, default=49000, metavar='N',
                        help='number of training examples')
    parser.add_argument('--n-val', type=int, default=1000, metavar='N',
                        help='number of validation examples')
    parser.add_argument('--data-path', type=str, default='../data/', metavar='PATH',
                        help='directory for storing augmented data etc.')

    # small RBMs related
    parser.add_argument('--small-lr', type=float, default=5e-4, metavar='LR', nargs='+',
                        help='learning rate or sequence of such (per epoch)')
    parser.add_argument('--small-epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--small-batch-size', type=int, default=48, metavar='B',
                        help='input batch size for training')
    parser.add_argument('--small-l2', type=float, default=1e-3, metavar='L2',
                        help='L2 weight decay coefficient')
    parser.add_argument('--small-dirpath-prefix', type=str, default='../models/rbm_small_', metavar='PREFIX',
                        help='directory path prefix to save RBMs trained on patches')



    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


    # prepare data (load + scale + split)
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
    X_aug = None
    X_aug_path = os.path.join(args.data_path, 'X.npy')
    augment = True
    if os.path.isfile(X_aug_path):
        print "\nLoading augmented data ..."
        X_aug = np.load(X_aug_path)
        print "Checking augmented data ..."
        if 10 * n_train == len(X_aug):
            augment = False

    if augment:
        print "\nAugmenting data ..."
        s = Stopwatch(verbose=True).start()

        X_aug = np.zeros((10 * n_train, 32, 32, 3), dtype=np.float32)
        X_train = im_unflatten(X_train)
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
        X_aug = im_flatten(X_aug)

        # save to disk
        np.save(X_aug_path, X_aug)

        s.elapsed()
        print "\n"

    # convert + scale augmented data again
    X_train = X_aug.astype(np.float32)
    X_train /= 255.
    print "Augmented shape: {0}".format(X_train.shape)
    print "Augmented range: {0}".format((X_train.min(), X_train.max()))

    # center and normalize training data
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_train -= X_mean
    X_train /= X_std
    np.save(os.path.join(args.data_path, 'X_mean.npy'), X_mean)
    np.save(os.path.join(args.data_path, 'X_std.npy'), X_std)
    print "Augmented mean: ({0:.3f}, ...); std: ({1:.3f}, ...)".format(X_train.mean(axis=0)[0],
                                                                       X_train.std(axis=0)[0])
    print "Augmented range: ({0:.3f}, {1:.3f})\n\n".format(X_train.min(), X_train.max())


    # train 26 small Gaussian RBMs on patches
    X_train = im_unflatten(X_train)
    X_patches = X_train[:, :8, :8, :]
    X_patches = im_flatten(X_patches)
    print X_patches.shape

    rbm = GaussianRBM(n_visible=8*8*3,
                      n_hidden=300,
                      sigma=1.,
                      W_init=0.001,
                      vb_init=0.,
                      hb_init=0.,
                      n_gibbs_steps=1,
                      learning_rate=args.small_lr,
                      momentum=np.geomspace(0.5, 0.9, 8),
                      max_epoch=args.small_epochs,
                      batch_size=args.small_batch_size,
                      l2=args.small_l2,
                      sample_v_states=True,
                      sample_h_states=True,
                      sparsity_target=0.1,
                      sparsity_cost=1e-4,
                      metrics_config=dict(
                          msre=True,
                          feg=True,
                          train_metrics_every_iter=100,
                          val_metrics_every_epoch=1,
                          feg_every_epoch=2,
                          n_batches_for_feg=50,
                      ),
                      verbose=True,
                      display_filters=12,
                      v_shape=(8, 8, 3),
                      display_hidden_activations=24,
                      random_seed=9000 + 0,
                      tf_dtype='float32',
                      tf_saver_params=dict(max_to_keep=1),
                      model_path=args.small_dirpath_prefix + '0/')
    rbm.fit(X_patches)


if __name__ == '__main__':
    main()
