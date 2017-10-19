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

import env
from hdm.dbm import DBM
from hdm.rbm import GaussianRBM, MultinomialRBM
from hdm.utils import RNG, Stopwatch
from hdm.utils.augmentation import shift, horizontal_mirror
from hdm.utils.dataset import (load_cifar10,
                               im_flatten, im_unflatten)


def make_augmentation(X_train, n_train, args):
    X_aug = None
    X_aug_path = os.path.join(args.data_path, 'X_aug.npy')
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
                    (1, 0),
                    (-1, 0),
                    (0, 1),
                    (0, -1)
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

    return X_aug

def make_small_rbms(X_train, X_val, small_rbm_config, args):
    X_train = im_unflatten(X_train)
    X_val = im_unflatten(X_val)
    small_rbms = []

    # first 16 ...
    for i in xrange(4):
        for j in xrange(4):
            rbm_id = 4 * i + j
            rbm_dirpath = args.small_dirpath_prefix + str(rbm_id) + '/'

            if os.path.isdir(rbm_dirpath):
                print "\nLoading small RBM #{0} ...\n\n".format(rbm_id)
                rbm = GaussianRBM.load_model(rbm_dirpath)
            else:
                print "\nTraining small RBM #{0} ...\n\n".format(rbm_id)
                X_patches = X_train[:, 8 * i:8 * (i + 1),
                            8 * j:8 * (j + 1), :]
                X_patches_val = X_val[:, 8 * i:8 * (i + 1),
                                8 * j:8 * (j + 1), :]
                X_patches = im_flatten(X_patches)
                X_patches_val = im_flatten(X_patches_val)

                rbm = GaussianRBM(random_seed=9000 + rbm_id,
                                  model_path=rbm_dirpath,
                                  **small_rbm_config)
                rbm.fit(X_patches, X_patches_val)
            small_rbms.append(rbm)

    # next 9 ...
    for i in xrange(3):
        for j in xrange(3):
            rbm_id = 16 + 3 * i + j
            rbm_dirpath = args.small_dirpath_prefix + str(rbm_id) + '/'

            if os.path.isdir(rbm_dirpath):
                print "\nLoading small RBM #{0} ...\n\n".format(rbm_id)
                rbm = GaussianRBM.load_model(rbm_dirpath)
            else:
                print "\nTraining small RBM #{0} ...\n\n".format(rbm_id)
                X_patches = X_train[:, 4 + 8 * i:4 + 8 * (i + 1),
                            4 + 8 * j:4 + 8 * (j + 1), :]
                X_patches_val = X_val[:, 4 + 8 * i:4 + 8 * (i + 1),
                                4 + 8 * j:4 + 8 * (j + 1), :]
                X_patches = im_flatten(X_patches)
                X_patches_val = im_flatten(X_patches_val)

                rbm = GaussianRBM(random_seed=9000 + rbm_id,
                                  model_path=rbm_dirpath,
                                  **small_rbm_config)
                rbm.fit(X_patches, X_patches_val)
            small_rbms.append(rbm)

    # ... the last one
    rbm_id = 25
    rbm_dirpath = args.small_dirpath_prefix + str(rbm_id) + '/'

    if os.path.isdir(rbm_dirpath):
        print "\nLoading small RBM #{0} ...\n\n".format(rbm_id)
        rbm = GaussianRBM.load_model(rbm_dirpath)
    else:
        print "\nTraining small RBM #{0} ...\n\n".format(rbm_id)
        X_patches = X_train.copy()  # (N, 32, 32, 3)
        X_patches = X_patches.transpose(0, 3, 1, 2)  # (N, 3, 32, 32)
        X_patches = X_patches.reshape((-1, 3, 4, 8, 4, 8)).mean(axis=4).mean(axis=2)  # (N, 3, 8, 8)
        X_patches = X_patches.transpose(0, 2, 3, 1)  # (N, 8, 8, 3)
        X_patches = im_flatten(X_patches)  # (N, 8*8*3)

        X_patches_val = X_val.copy()
        X_patches_val = X_patches_val.transpose(0, 3, 1, 2)
        X_patches_val = X_patches_val.reshape((-1, 3, 4, 8, 4, 8)).mean(axis=4).mean(axis=2)
        X_patches_val = X_patches_val.transpose(0, 2, 3, 1)
        X_patches_val = im_flatten(X_patches_val)

        rbm = GaussianRBM(random_seed=9000 + rbm_id,
                          model_path=rbm_dirpath,
                          **small_rbm_config)
        rbm.fit(X_patches, X_patches_val)
    small_rbms.append(rbm)
    return small_rbms

def make_large_weights(small_rbms):
    W = np.zeros((300 * 26, 32, 32, 3), dtype=np.float32)
    W[...] = RNG(seed=1234).rand(*W.shape) * 5e-6
    vb = np.zeros((32, 32, 3))
    hb = np.zeros(300 * 26)

    for i in xrange(4):
        for j in xrange(4):
            rbm_id = 4 * i + j
            weights = small_rbms[rbm_id].get_tf_params(scope='weights')
            W_small = weights['W']
            W_small = W_small.T  # (300, 192)
            W_small = im_unflatten(W_small)  # (300, 8, 8, 3)
            W[300 * rbm_id: 300 * (rbm_id + 1), 8 * i:8 * (i + 1),
                                                8 * j:8 * (j + 1), :] = W_small
            vb[8 * i:8 * (i + 1),
               8 * j:8 * (j + 1), :] += im_unflatten(weights['vb'])
            hb[300 * rbm_id: 300 * (rbm_id + 1)] = weights['hb']

    for i in xrange(3):
        for j in xrange(3):
            rbm_id = 16 + 3 * i + j
            weights = small_rbms[rbm_id].get_tf_params(scope='weights')
            W_small = weights['W']
            W_small = W_small.T
            W_small = im_unflatten(W_small)
            W[300 * rbm_id: 300 * (rbm_id + 1), 4 + 8 * i:4 + 8 * (i + 1),
                                                4 + 8 * j:4 + 8 * (j + 1), :] = W_small
            vb[4 + 8 * i:4 + 8 * (i + 1),
               4 + 8 * j:4 + 8 * (j + 1), :] += im_unflatten(weights['vb'])
            hb[300 * rbm_id: 300 * (rbm_id + 1)] = weights['hb']

    weights = small_rbms[25].get_tf_params(scope='weights')
    W_small = weights['W']
    W_small = W_small.T
    W_small = im_unflatten(W_small)
    vb_small = im_unflatten(weights['vb'])
    for i in xrange(8):
        for j in xrange(8):
            U = W_small[:, i, j, :]
            U = np.expand_dims(U, -1)
            U = np.expand_dims(U, -1)
            U = U.transpose(0, 2, 3, 1)
            W[-300:, 4 * i:4 * (i + 1),
                     4 * j:4 * (j + 1), :] = U / 16.
            vb[4 * i:4 * (i + 1),
               4 * j:4 * (j + 1), :] += vb_small[i, j, :].reshape((1, 1, 3)) / 16.
            hb[-300:] = weights['hb']

    W = im_flatten(W)
    W = W.T
    vb /= 2.
    vb[4:-4, 4:-4, :] /= 1.5
    vb = im_flatten(vb)

    return W, vb, hb

def make_transform(rbm, X, path):
    H = None
    transform = True
    if os.path.isfile(path):
        H = np.load(path)
        if len(X) == len(H):
            transform = False
    if transform:
        H = rbm.transform(X).astype('float32')
        np.save(path, H)
    return H


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
    parser.add_argument('--small-lr', type=float, default=1e-3, metavar='LR', nargs='+',
                        help='learning rate or sequence of such (per epoch)')
    parser.add_argument('--small-epochs', type=int, default=120, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--small-batch-size', type=int, default=48, metavar='B',
                        help='input batch size for training')
    parser.add_argument('--small-l2', type=float, default=2e-3, metavar='L2',
                        help='L2 weight decay coefficient')
    parser.add_argument('--small-dirpath-prefix', type=str, default='../models/rbm_cifar_small_', metavar='PREFIX',
                        help='directory path prefix to save RBMs trained on patches')

    # RBM #2 related
    parser.add_argument('--increase-n-gibbs-steps-every', type=int, default=16, metavar='I',
                        help='increase number of Gibbs steps every specified number of epochs for RBM #2')

    # common for RBMs and DBM
    parser.add_argument('--lr', type=float, default=[5e-4, 5e-5, 1e-4], metavar='LR', nargs='+',
                        help='(initial) learning rates')
    parser.add_argument('--epochs', type=int, default=[72, 80, 200], metavar='N', nargs='+',
                        help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=[100, 100, 100], metavar='B', nargs='+',
                        help='input batch size for training, `--n-train` and `--n-val`' + \
                             'must be divisible by this number (for DBM)')
    parser.add_argument('--l2', type=float, default=[2e-3, 0.005, 1e-7], metavar='L2', nargs='+',
                        help='L2 weight decay coefficient')

    # save dirpaths
    parser.add_argument('--rbm1-dirpath', type=str, default='../models/rbm1_cifar/', metavar='DIRPATH',
                        help='directory path to save Gaussian RBM')
    parser.add_argument('--rbm2-dirpath', type=str, default='../models/rbm2_cifar/', metavar='DIRPATH',
                        help='directory path to save Multinomial RBM')
    parser.add_argument('--dbm-dirpath', type=str, default='../models/dbm_cifar/', metavar='DIRPATH',
                        help='directory path to save DBM')

    # DBM related
    parser.add_argument('--n-particles', type=int, default=100, metavar='M',
                        help='number of persistent Markov chains')
    parser.add_argument('--n-gibbs-steps', type=int, default=1, metavar='N',
                        help='number of Gibbs steps for PCD')
    parser.add_argument('--max-mf-updates', type=int, default=50, metavar='N',
                        help='maximum number of mean-field updates per weight update')
    parser.add_argument('--mf-tol', type=float, default=1e-7, metavar='TOL',
                        help='mean-field tolerance')
    parser.add_argument('--max-norm', type=float, default=8., metavar='C',
                        help='maximum norm constraint')
    parser.add_argument('--sparsity-target', type=float, default=[0.2, 0.1], metavar='T', nargs='+',
                        help='desired probability of hidden activation')
    parser.add_argument('--sparsity-cost', type=float, default=[1e-5, 1e-5], metavar='C', nargs='+',
                        help='controls the amount of sparsity penalty')
    parser.add_argument('--sparsity-damping', type=float, default=0.9, metavar='D',
                        help='decay rate for hidden activations probs')

    # parse and check params
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if len(args.lr) == 1: args.lr *= 3
    if len(args.epochs) == 1: args.epochs *= 3
    if len(args.batch_size) == 1: args.batch_size *= 3
    if len(args.l2) == 1: args.l2 *= 3

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

    # augment data
    X_aug = make_augmentation(X_train, n_train, args)

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
    X_val -= X_mean
    X_val /= X_std
    mean_path = os.path.join(args.data_path, 'X_mean.npy')
    std_path = os.path.join(args.data_path, 'X_std.npy')
    if not os.path.isfile(mean_path): np.save(mean_path, X_mean)
    if not os.path.isfile(std_path): np.save(std_path, X_std)
    print "Augmented mean: ({0:.3f}, ...); std: ({1:.3f}, ...)".format(X_train.mean(axis=0)[0],
                                                                       X_train.std(axis=0)[0])
    print "Augmented range: ({0:.3f}, {1:.3f})\n\n".format(X_train.min(), X_train.max())

    # train 26 small Gaussian RBMs on patches
    small_rbm_config = dict(n_visible=8 * 8 * 3,
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
                            v_shape=(8, 8, 3),
                            display_hidden_activations=36,
                            tf_dtype='float32',
                            tf_saver_params=dict(max_to_keep=1))
    small_rbms = make_small_rbms(X_train, X_val, small_rbm_config, args)

    # assemble large weight matrix and biases
    print "\nAssembling weights for large Gaussian RBM ...\n\n"
    W, vb, hb = make_large_weights( small_rbms )

    # pre-train large Gaussian RBM (G-RBM)
    if os.path.isdir(args.rbm1_dirpath):
        print "\nLoading G-RBM ...\n\n"
        grbm = GaussianRBM.load_model(args.rbm1_dirpath)
    else:
        print "\nTraining G-RBM ...\n\n"
        grbm = GaussianRBM(n_visible=32 * 32 * 3,
                           n_hidden=300 * 26,
                           sigma=1.,
                           W_init=W,
                           vb_init=vb,
                           hb_init=hb,
                           n_gibbs_steps=1,
                           learning_rate=args.lr[0],
                           momentum=np.geomspace(0.5, 0.9, 8),
                           max_epoch=args.epochs[0],
                           batch_size=args.batch_size[0],
                           l2=args.l2[0],
                           sample_v_states=True,
                           sample_h_states=True,
                           sparsity_target=0.1,
                           sparsity_cost=1e-4,
                           dbm_first=True,  # !!!
                           metrics_config=dict(
                               msre=True,
                               feg=True,
                               train_metrics_every_iter=1000,
                               val_metrics_every_epoch=1,
                               feg_every_epoch=2,
                               n_batches_for_feg=50,
                           ),
                           verbose=True,
                           display_filters=24,
                           v_shape=(32, 32, 3),
                           display_hidden_activations=36,
                           random_seed=1111,
                           tf_dtype='float32',
                           tf_saver_params=dict(max_to_keep=1),
                           model_path=args.rbm1_dirpath)
        grbm.fit(X_train, X_val)

    # extract features Q = P_{G-RBM}(h|v=X)
    print "\nExtracting features from G-RBM ...\n\n"
    Q_train_path = os.path.join(args.data_path, 'Q_train_cifar.npy')
    Q_val_path = os.path.join(args.data_path, 'Q_val_cifar.npy')
    Q_train = make_transform(grbm, X_train, Q_train_path)
    Q_val = make_transform(grbm, X_val, Q_val_path)

    # pre-train Multinomial RBM (M-RBM)
    if os.path.isdir(args.rbm2_dirpath):
        print "\nLoading M-RBM ...\n\n"
        mrbm = MultinomialRBM.load_model(args.rbm2_dirpath)
    else:
        print "\nTraining M-RBM ...\n\n"
        mrbm_lr = args.lr[1]
        mrbm_config = dict(n_visible=300 * 26,
                           n_hidden=512,
                           n_samples=512,
                           W_init=0.001,
                           hb_init=0.,
                           vb_init=0.,
                           n_gibbs_steps=1,
                           learning_rate=mrbm_lr,
                           momentum=np.geomspace(0.5, 0.9, 8),
                           max_epoch=args.increase_n_gibbs_steps_every,
                           batch_size=args.batch_size[1],
                           l2=args.l2[1],
                           sample_h_states=True,
                           sample_v_states=True,
                           sparsity_target=0.2,
                           sparsity_cost=1e-4,
                           dbm_last=True,  # !!!
                           metrics_config=dict(
                               msre=True,
                               pll=True,
                               feg=True,
                               train_metrics_every_iter=5,
                               val_metrics_every_epoch=1,
                               feg_every_epoch=2,
                               n_batches_for_feg=50,
                           ),
                           verbose=True,
                           display_hidden_activations=100,
                           random_seed=2222,
                           tf_dtype='float32',
                           tf_saver_params=dict(max_to_keep=1),
                           model_path=args.rbm2_dirpath)
        max_epoch = args.increase_n_gibbs_steps_every
        mrbm = MultinomialRBM(**mrbm_config)
        mrbm.fit(Q_train, Q_val)
        mrbm_config['momentum'] = 0.9
        while max_epoch < args.epochs[1]:
            max_epoch += args.increase_n_gibbs_steps_every
            max_epoch = min(max_epoch, args.epochs[1])
            mrbm_config['max_epoch'] = max_epoch
            mrbm_config['n_gibbs_steps'] += 1
            mrbm_config['learning_rate'] = mrbm_lr / float(mrbm_config['n_gibbs_steps'])

            print "\nNumber of Gibbs steps = {0}, learning rate = {1:.6f} ...\n\n". \
                format(mrbm_config['n_gibbs_steps'], mrbm_config['learning_rate'])

            mrbm_new = MultinomialRBM(**mrbm_config)
            mrbm_new.init_from(mrbm)
            mrbm = mrbm_new
            mrbm.fit(Q_train, Q_val)

    # extract features G = P_{M-RBM}(h|v=Q)
    print "\nExtracting features from M-RBM ...\n\n"
    G_train_path = os.path.join(args.data_path, 'G_train_cifar.npy')
    G_val_path = os.path.join(args.data_path, 'G_val_cifar.npy')
    G_train = make_transform(grbm, Q_train, G_train_path)
    G_val = make_transform(grbm, Q_val, G_val_path)

    print G_train.shape, G_val.shape

    print "\nTraining DBM ...\n\n"
    dbm = DBM(rbms=[grbm, mrbm],
              n_particles=args.n_particles,
              v_particle_init=X_train[:args.n_particles].copy(),
              h_particles_init=(Q_train[:args.n_particles].copy(),
                                G_train[:args.n_particles].copy()),
              n_gibbs_steps=args.n_gibbs_steps,
              max_mf_updates=args.max_mf_updates,
              mf_tol=args.mf_tol,
              learning_rate=np.geomspace(args.lr[2], 5e-6, args.epochs[2]),
              momentum=np.geomspace(0.5, 0.9, 10),
              max_epoch=args.epochs[2],
              batch_size=args.batch_size[2],
              l2=args.l2[2],
              max_norm=args.max_norm,
              sample_v_states=True,
              sample_h_states=(True, True),
              sparsity_targets=args.sparsity_target,
              sparsity_costs=args.sparsity_cost,
              sparsity_damping=args.sparsity_damping,
              train_metrics_every_iter=400,
              val_metrics_every_epoch=2,
              random_seed=3333,
              verbose=True,
              display_filters=12,
              v_shape=(32, 32, 3),
              display_particles=36,
              tf_dtype='float32',
              tf_saver_params=dict(max_to_keep=1),
              model_path=args.dbm_dirpath)
    dbm.fit(X_train, X_val)


if __name__ == '__main__':
    main()
