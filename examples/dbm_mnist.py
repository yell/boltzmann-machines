#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train 2-layer Bernoulli DBM on MNIST dataset.
Hyper-parameters are similar to those in MATLAB code [1].
RBM #2 trained with increasing k in CD-k and decreasing learning rate.

Links
-----
[1] http://www.cs.toronto.edu/~rsalakhu/DBM.html
"""
print __doc__


import argparse
import numpy as np

import env
from hdm.dbm import DBM
from hdm.rbm import BernoulliRBM
from hdm.utils import RNG
from hdm.utils.dataset import load_mnist


def main():
    # training settings
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n-train', type=int, default=55000, metavar='N',
                        help='number of training examples')
    parser.add_argument('--n-val', type=int, default=5000, metavar='N',
                        help='number of validation examples')

    parser.add_argument('--n-hiddens', type=int, default=[512, 1024], metavar='N', nargs='+',
                        help='numbers of hidden units')
    parser.add_argument('--epochs', type=int, default=[64, 144, 300], metavar='N', nargs='+',
                        help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=48, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--increase-n-gibbs-steps-every', type=int, default=20, metavar='I',
                        help='increase number of Gibbs steps every specified number of epochs for RBM #2')

    parser.add_argument('--rbm1-dirpath', type=str, default='../models/dbm_mnist_rbm1/', metavar='DIRPATH',
                        help='directory path to save RBM #1')
    parser.add_argument('--rbm2-dirpath', type=str, default='../models/dbm_mnist_rbm2/', metavar='DIRPATH',
                        help='directory path to save RBM #2')
    parser.add_argument('--dbm-dirpath', type=str, default='../models/dbm_mnist/', metavar='DIRPATH',
                        help='directory path to save DBM')

    parser.add_argument('--load-rbm1', type=str, default=None, metavar='DIRPATH',
                        help='directory path to load trained RBM #1')
    parser.add_argument('--load-rbm2', type=str, default=None, metavar='DIRPATH',
                        help='directory path to load trained RBM #2')
    args = parser.parse_args()

    # prepare data
    X, _ = load_mnist(mode='train', path='../data/')
    X /= 255.
    RNG(seed=42).shuffle(X)
    X_train = X[:args.n_train]
    X_val = X[-args.n_val:]
    X = np.concatenate((X_train, X_val))

    # pre-train RBM #1
    if args.load_rbm1:
        print "\nLoading RBM #1 ...\n\n"
        rbm1 = BernoulliRBM.load_model(args.load_rbm1)
    else:
        print "\nTraining RBM #1 ...\n\n"
        rbm1 = BernoulliRBM(n_visible=784,
                            n_hidden=args.n_hiddens[0],
                            W_init=0.001,
                            vb_init=0.,
                            hb_init=0.,
                            n_gibbs_steps=1,
                            learning_rate=0.05,
                            momentum=[0.5] * 5 + [0.9],
                            max_epoch=args.epochs[0],
                            batch_size=args.batch_size,
                            L2=1e-3,
                            sample_h_states=True,
                            sample_v_states=True,
                            sparsity_cost=0.,
                            dbm_first=True, # !!!
                            metrics_config=dict(
                                msre=True,
                                pll=True,
                                train_metrics_every_iter=500,
                            ),
                            verbose=True,
                            random_seed=1337,
                            tf_dtype='float32',
                            tf_saver_params=dict(max_to_keep=1),
                            model_path=args.rbm1_dirpath)
        rbm1.fit(X)

    # freeze RBM #1 and extract features P(h|data)
    print "\nExtracting features from RBM #1 ...\n\n"
    H = rbm1.transform(X)
    print H.shape

    # pre-train RBM #2
    if args.load_rbm2:
        print "\nLoading RBM #2 ...\n\n"
        rbm2 = BernoulliRBM.load_model(args.load_rbm2)
    else:
        print "\nTraining RBM #2 ...\n\n"
        rbm2_config = dict(
            n_visible=args.n_hiddens[0],
            n_hidden=args.n_hiddens[1],
            W_init=0.01,
            vb_init=0.,
            hb_init=0.,
            n_gibbs_steps=1,
            learning_rate=0.05,
            momentum=[0.5] * 5 + [0.9],
            max_epoch=args.increase_n_gibbs_steps_every,
            batch_size=args.batch_size,
            L2=1e-3,
            sample_h_states=True,
            sample_v_states=True,
            sparsity_cost=0.,
            dbm_last=True,  # !!!
            metrics_config=dict(
                msre=True,
                pll=True,
                train_metrics_every_iter=100,
            ),
            verbose=True,
            display_filters=False,
            random_seed=9000,
            tf_dtype='float32',
            tf_saver_params=dict(max_to_keep=1),
            model_path=args.rbm2_dirpath
        )
        max_epoch = args.increase_n_gibbs_steps_every
        rbm2 = BernoulliRBM(**rbm2_config)
        rbm2.fit(H)
        rbm2_config['momentum'] = 0.9
        while max_epoch < args.epochs[1]:
            print "\nIncreasing number of Gibbs steps, decreasing learning rate ...\n\n"
            max_epoch += args.increase_n_gibbs_steps_every
            max_epoch = min(max_epoch, args.epochs[1])
            rbm2_config['max_epoch'] = max_epoch
            rbm2_config['n_gibbs_steps'] += 1
            rbm2_config['learning_rate'] = 0.05/float(rbm2_config['n_gibbs_steps'])
            rbm2_new = BernoulliRBM(**rbm2_config)
            rbm2_new.init_from(rbm2)
            rbm2 = rbm2_new
            rbm2.fit(H)

    # DBM <- (rbm1, rbm2)


if __name__ == '__main__':
    main()