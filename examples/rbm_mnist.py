#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train Bernoulli-Bernoulli RBM on MNIST dataset.

Momentum is initially 0.5 and gradually increases to 0.9.
Training time is approx. 2.5 times faster using single-precision rather than double
with negligible difference in reconstruction error, pseudo log-lik is slightly more noisy
at the beginning of training though.
"""
print __doc__


import argparse
import numpy as np

import env
from hdm.rbm import BernoulliRBM, logit_mean
from hdm.utils import RNG
from hdm.utils.dataset import load_mnist


def main():
    # training settings
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n-train', type=int, default=55000, metavar='N',
                        help='number of training examples')
    parser.add_argument('--n-val', type=int, default=5000, metavar='N',
                        help='number of validation examples')
    parser.add_argument('--n-hidden', type=int, default=1024, metavar='N',
                        help='number of hidden units')
    parser.add_argument('--vb-init', action='store_false',
                        help='initialize visible biases as logit of mean values of features'+\
                             ', otherwise zero init')
    parser.add_argument('--hb-init', type=float, default=0., metavar='HB',
                        help='initial hidden bias')
    parser.add_argument('--n-gibbs-steps', type=int, default=1, metavar='N',
                        help='number of Gibbs updates per iteration')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR', nargs='+',
                        help='learning rate(s)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--l2', type=float, default=1e-5, metavar='L2',
                        help='L2 weight decay coefficient')
    parser.add_argument('--sample-v-states', action='store_true',
                        help='sample visible states, otherwise use probabilities w/o sampling')
    parser.add_argument('--dropout', type=float, metavar='P',
                        help='probability of visible units being on')
    parser.add_argument('--sparsity-target', type=float, default=0.1, metavar='T',
                        help='desired probability of hidden activation')
    parser.add_argument('--sparsity-cost', type=float, default=1e-5, metavar='C',
                        help='controls the amount of sparsity penalty')
    parser.add_argument('--sparsity-damping', type=float, default=0.9, metavar='D',
                        help='decay rate for hidden activations probs')
    parser.add_argument('--dtype', type=str, default='float32', metavar='T',
                        help="datatype precision to use, {'float32', 'float64'}")
    parser.add_argument('--model-dirpath', type=str, default='../models/rbm_mnist/', metavar='DIRPATH',
                        help='directory path to save the model')
    args = parser.parse_args()

    # prepare data
    X, _ = load_mnist(mode='train', path='../data/')
    X /= 255.
    RNG(seed=42).shuffle(X)
    X_train = X[:args.n_train]
    X_val = X[-args.n_val:]

    # train and save the model
    rbm = BernoulliRBM(n_visible=784,
                       n_hidden=args.n_hidden,
                       vb_init=logit_mean(X_train) if args.vb_init else 0.,
                       hb_init=args.hb_init,
                       n_gibbs_steps=args.n_gibbs_steps,
                       learning_rate=args.lr,
                       momentum=np.geomspace(0.5, 0.9, 8),
                       max_epoch=args.epochs,
                       batch_size=args.batch_size,
                       L2=args.l2,
                       sample_h_states=True,
                       sample_v_states=args.sample_v_states,
                       dropout=args.dropout,
                       sparsity_target=args.sparsity_target,
                       sparsity_cost=args.sparsity_cost,
                       sparsity_damping=args.sparsity_damping,
                       metrics_config=dict(
                           msre=True,
                           pll=True,
                           feg=True,
                           train_metrics_every_iter=1000,
                           val_metrics_every_epoch=2,
                           feg_every_epoch=4,
                           n_batches_for_feg=50,
                       ),
                       verbose=True,
                       random_seed=1337,
                       tf_dtype=args.dtype,
                       tf_saver_params=dict(max_to_keep=1),
                       model_path=args.model_dirpath)
    rbm.fit(X_train, X_val)


if __name__ == '__main__':
    main()
