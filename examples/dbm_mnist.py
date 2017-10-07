#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train 2-layer Bernoulli DBM on MNIST dataset.
Hyper-parameters are mostly from MATLAB code [1].

Links
-----
[1] http://www.cs.toronto.edu/~rsalakhu/DBM.html
"""
print __doc__


import argparse

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
    parser.add_argument('--batch-size', type=int, default=48, metavar='N',
                        help='input batch size for training')

    parser.add_argument('--load-rbm1', type=str, default=None, metavar='DIRPATH',
                        help='directory path to load pre-trained RBM #1')
    parser.add_argument('--load-rbm2', type=str, default=None, metavar='DIRPATH',
                        help='directory path to load pre-trained RBM #2')
    args = parser.parse_args()

    # prepare data
    X, _ = load_mnist(mode='train', path='../data/')
    X /= 255.
    RNG(seed=42).shuffle(X)
    X_train = X[:args.n_train]
    X_val = X[-args.n_val:]

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
                            learning_rate=0.01,
                            momentum=[.5] * 5 + [.9],
                            max_epoch=100,
                            batch_size=args.batch_size,
                            L2=1e-3,
                            sample_h_states=True,
                            sample_v_states=True,
                            sparsity_cost=0.,
                            dbm_first=True, # !!!
                            metrics_config=dict(
                                msre=True,
                                pll=True,
                                train_metrics_every_iter=200,
                            ),
                            verbose=True,
                            random_seed=1337,
                            tf_dtype='float32',
                            model_path='../models/dbm_mnist_rbm_1/')
        rbm1.fit(X_train)

    # RBM2 = [1] * 20 -> [2] * 20 -> ...


if __name__ == '__main__':
    main()