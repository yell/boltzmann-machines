#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train 3072-5000-1000 Gaussian-Bernoulli-Multinomial
DBM with pre-training on "smoothed" CIFAR-10 (with 1000 least
significant singular values removed), as suggested in [1].

The training took approx. 47 + 119 + C min = Z on GTX 1060.

References
----------
[1] A. Krizhevsky and G. Hinton. Learning multiple layers of features
    from tine images. 2009.
"""
print __doc__


import os
import argparse
import numpy as np
from scipy.linalg import svd

import env
from bm import DBM
from bm.rbm import GaussianRBM, MultinomialRBM
from bm.utils import RNG, Stopwatch
from bm.utils.dataset import load_cifar10


def make_smoothing(X_train, n_train, args):
    X_s = None
    X_s_path = os.path.join(args.data_path, 'X_s.npy')

    do_smoothing = True
    if os.path.isfile(X_s_path):
        print "\nLoading smoothed data ..."
        X_s = np.load(X_s_path)
        print "Checking augmented data ..."
        if len(X_s) == n_train:
            do_smoothing = False

    if do_smoothing:
        print "\nSmoothing data ..."
        X_m = X_train.mean(axis=0)
        X_train -= X_m
        with Stopwatch(verbose=True) as s:
            [U, s, Vh] = svd(X_train,
                             full_matrices=False,
                             compute_uv=True,
                             overwrite_a=True,
                             check_finite=False)
            s[-1000:] = 0.
            X_s = U.dot(np.diag(s).dot(Vh))
            X_s += X_m

        # save to disk
        np.save(X_s_path, X_s)
        print "\n"

    return X_s

def make_grbm((X_train, X_val), args):
    if os.path.isdir(args.grbm_dirpath):
        print "\nLoading G-RBM ...\n\n"
        grbm = GaussianRBM.load_model(args.grbm_dirpath)
    else:
        print "\nTraining G-RBM ...\n\n"
        grbm = GaussianRBM(n_visible=32 * 32 * 3,
                           n_hidden=5000,
                           sigma=1.,
                           W_init=0.0008,
                           vb_init=0.,
                           hb_init=0.,
                           n_gibbs_steps=args.n_gibbs_steps[0],
                           learning_rate=args.lr[0],
                           momentum=np.geomspace(0.5, 0.9, 8),
                           max_epoch=args.epochs[0],
                           batch_size=args.batch_size[0],
                           l2=args.l2[0],
                           sample_v_states=True,
                           sample_h_states=True,
                           sparsity_cost=0.,
                           dbm_first=True,  # !!!
                           metrics_config=dict(
                               msre=True,
                               feg=True,
                               train_metrics_every_iter=1000,
                               val_metrics_every_epoch=2,
                               feg_every_epoch=2,
                               n_batches_for_feg=50,
                           ),
                           verbose=True,
                           display_filters=12,
                           display_hidden_activations=24,
                           v_shape=(32, 32, 3),
                           dtype='float32',
                           tf_saver_params=dict(max_to_keep=1),
                           model_path=args.grbm_dirpath)
        grbm.fit(X_train, X_val)
    return grbm

def make_mrbm((Q_train, Q_val), args):
    if os.path.isdir(args.mrbm_dirpath):
        print "\nLoading M-RBM ...\n\n"
        mrbm = MultinomialRBM.load_model(args.mrbm_dirpath)
    else:
        print "\nTraining M-RBM ...\n\n"
        mrbm = MultinomialRBM(n_visible=5000,
                              n_hidden=1000,
                              n_samples=1000,
                              W_init=0.01,
                              hb_init=0.,
                              vb_init=0.,
                              n_gibbs_steps=args.n_gibbs_steps[1],
                              learning_rate=args.lr[1],
                              momentum=np.geomspace(0.5, 0.9, 8),
                              max_epoch=args.epochs[1],
                              batch_size=args.batch_size[1],
                              l2=args.l2[1],
                              sample_h_states=True,
                              sample_v_states=False,
                              sparsity_cost=0.,
                              dbm_last=True,  # !!!
                              metrics_config=dict(
                                  msre=True,
                                  pll=True,
                                  feg=True,
                                  train_metrics_every_iter=400,
                                  val_metrics_every_epoch=2,
                                  feg_every_epoch=2,
                                  n_batches_for_feg=50,
                              ),
                              verbose=True,
                              display_filters=0,
                              display_hidden_activations=100,
                              random_seed=1337,
                              dtype='float32',
                              tf_saver_params=dict(max_to_keep=1),
                              model_path=args.mrbm_dirpath)
        mrbm.fit(Q_train, Q_val)
    return mrbm

def make_rbm_transform(rbm, X, path, np_dtype=None):
    H = None
    transform = True
    if os.path.isfile(path):
        H = np.load(path)
        if len(X) == len(H):
            transform = False
    if transform:
        H = rbm.transform(X, np_dtype=np_dtype)
        np.save(path, H)
    return H

def make_dbm((X_train, X_val), rbms, (Q, G), args):
    if os.path.isdir(args.dbm_dirpath):
        print "\nLoading DBM ...\n\n"
        dbm = DBM.load_model(args.dbm_dirpath)
        dbm.load_rbms(rbms)  # !!!
    else:
        print "\nTraining DBM ...\n\n"
        dbm = DBM(rbms=rbms,
                  n_particles=args.n_particles,
                  v_particle_init=X_train[:args.n_particles].copy(),
                  h_particles_init=(Q[:args.n_particles].copy(),
                                    G[:args.n_particles].copy()),
                  n_gibbs_steps=args.n_gibbs_steps[2],
                  max_mf_updates=args.max_mf_updates,
                  mf_tol=args.mf_tol,
                  learning_rate=np.geomspace(args.lr[2], 1e-5, args.epochs[2]),
                  momentum=np.geomspace(0.5, 0.9, 10),
                  max_epoch=args.epochs[2],
                  batch_size=args.batch_size[2],
                  l2=args.l2[2],
                  max_norm=args.max_norm,
                  sample_v_states=True,
                  sample_h_states=(True, True),
                  sparsity_cost=0.,
                  train_metrics_every_iter=1000,
                  val_metrics_every_epoch=2,
                  random_seed=args.random_seed[2],
                  verbose=True,
                  save_after_each_epoch=True,
                  display_filters=12,
                  display_particles=36,
                  v_shape=(32, 32, 3),
                  dtype='float32',
                  tf_saver_params=dict(max_to_keep=1),
                  model_path=args.dbm_dirpath)
        dbm.fit(X_train, X_val)
    return dbm


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

    # common for RBMs and DBM
    parser.add_argument('--n-gibbs-steps', type=int, default=(1, 1, 1), metavar='N', nargs='+',
                        help='(initial) number of Gibbs steps for CD/PCD')
    parser.add_argument('--lr', type=float, default=(5e-4, 1e-4, 8e-5), metavar='LR', nargs='+',
                        help='(initial) learning rates')
    parser.add_argument('--epochs', type=int, default=(120, 180, 1500), metavar='N', nargs='+',
                        help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=(100, 100, 100), metavar='B', nargs='+',
                        help='input batch size for training, `--n-train` and `--n-val`' + \
                             'must be divisible by this number (for DBM)')
    parser.add_argument('--l2', type=float, default=(0.01, 0.05, 1e-8), metavar='L2', nargs='+',
                        help='L2 weight decay coefficients')
    parser.add_argument('--random-seed', type=int, default=(1337, 1111, 2222), metavar='N', nargs='+',
                        help='random seeds for models training')

    # save dirpaths
    parser.add_argument('--grbm-dirpath', type=str, default='../models/grbm_cifar_naive/', metavar='DIRPATH',
                        help='directory path to save Gaussian RBM')
    parser.add_argument('--mrbm-dirpath', type=str, default='../models/mrbm_cifar_naive/', metavar='DIRPATH',
                        help='directory path to save Multinomial RBM')
    parser.add_argument('--dbm-dirpath', type=str, default='../models/dbm_cifar_naive/', metavar='DIRPATH',
                        help='directory path to save DBM')

    # DBM related
    parser.add_argument('--n-particles', type=int, default=100, metavar='M',
                        help='number of persistent Markov chains')
    parser.add_argument('--max-mf-updates', type=int, default=50, metavar='N',
                        help='maximum number of mean-field updates per weight update')
    parser.add_argument('--mf-tol', type=float, default=1e-11, metavar='TOL',
                        help='mean-field tolerance')
    parser.add_argument('--max-norm', type=float, default=4., metavar='C',
                        help='maximum norm constraint')

    # parse and check params
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    for x, m in (
            (args.n_gibbs_steps, 3),
            (args.lr, 3),
            (args.epochs, 3),
            (args.batch_size, 3),
            (args.l2, 3),
            (args.random_seed, 3),
    ):
        if len(x) == 1:
            x *= m

    # prepare data (load + scale + split)
    print "\nPreparing data ..."
    X, _ = load_cifar10(mode='train', path=args.data_path)
    X = X.astype(np.float32)
    X /= 255.
    RNG(seed=42).shuffle(X)
    n_train = min(len(X), args.n_train)
    n_val = min(len(X), args.n_val)
    X_train = X[:n_train]
    X_val = X[-n_val:]

    # remove 1000 least significant singular values
    X_train = make_smoothing(X_train, n_train, args)
    print X_train.shape

    # center and normalize training data
    X_s_mean = X_train.mean(axis=0)
    X_s_std = X_train.std(axis=0)
    mean_path = os.path.join(args.data_path, 'X_s_mean.npy')
    std_path = os.path.join(args.data_path, 'X_s_std.npy')
    if not os.path.isfile(mean_path):
        np.save(mean_path, X_s_mean)
    if not os.path.isfile(std_path):
        np.save(std_path, X_s_std)

    X_train -= X_s_mean
    X_train /= X_s_std
    X_val -= X_s_mean
    X_val /= X_s_std
    print "Mean: ({0:.3f}, ...); std: ({1:.3f}, ...)".format(X_train.mean(axis=0)[0],
                                                             X_train.std(axis=0)[0])
    print X_train[:10, :10]
    print "Range: ({0:.3f}, {1:.3f})\n\n".format(X_train.min(), X_train.max())

    # pre-train Gaussian RBM
    grbm = make_grbm((X_train, X_val), args)

    # extract features Q = p_{G-RBM}(h|v=X)
    print "\nExtracting features from G-RBM ...\n\n"
    Q_train, Q_val = None, None
    if not os.path.isdir(args.mrbm_dirpath) or not os.path.isdir(args.dbm_dirpath):
        Q_train_path = os.path.join(args.data_path, 'Q_train_cifar_naive.npy')
        Q_train = make_rbm_transform(grbm, X_train, Q_train_path, np_dtype=np.float16)
    if not os.path.isdir(args.mrbm_dirpath):
        Q_val_path = os.path.join(args.data_path, 'Q_val_cifar_naive.npy')
        Q_val = make_rbm_transform(grbm, X_val, Q_val_path)

    # pre-train Multinomial RBM (M-RBM)
    mrbm = make_mrbm((Q_train, Q_val), args)

    # extract features G = p_{M-RBM}(h|v=Q)
    print "\nExtracting features from M-RBM ...\n\n"
    G_train = None
    if not os.path.isdir(args.dbm_dirpath):
        G_train_path = os.path.join(args.data_path, 'G_train_cifar_naive.npy')
        G_train = make_rbm_transform(mrbm, Q_train, G_train_path)

    # jointly train DBM
    dbm = make_dbm((X_train, X_val), (grbm, mrbm), (Q_train, G_train), args)


if __name__ == '__main__':
    main()