#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train Bernoulli-Bernoulli RBM on MNIST dataset and use for classification.

Momentum is initially 0.5 and gradually increases to 0.9.
Training time is approx. 2.5 times faster using single-precision rather than double
with negligible difference in reconstruction error, pseudo log-lik is slightly more noisy
at the beginning of training though.
After the model is trained, it is discriminatively fine-tuned.
"""
print __doc__


import argparse
import numpy as np
from keras import regularizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.metrics import accuracy_score

import env
from hdm.rbm import BernoulliRBM, logit_mean
from hdm.utils import (RNG, Stopwatch,
                       one_hot, one_hot_decision_function, unhot)
from hdm.utils.dataset import load_mnist
from hdm.utils.optimizers import MultiAdam


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
                        help='learning rate or sequence of such (per epoch)')
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
    parser.add_argument('--load', type=str, default=None, metavar='DIRPATH',
                        help='directory path to load trained model')

    parser.add_argument('--mlp-no-init', action='store_true',
                        help='if enabled, use random initialization for MLP')
    parser.add_argument('--mlp-l2', type=float, default=1e-5, metavar='L2',
                        help='L2 weight decay coefficient for MLP')
    parser.add_argument('--mlp-lrm', type=float, default=(0.01, 1.), metavar='LRM', nargs='+',
                        help='learning rate multipliers of 1e-3 for MLP')
    parser.add_argument('--mlp-epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train MLP')
    parser.add_argument('--mlp-save-prefix', type=str, default='../data/rbm_', metavar='PREFIX',
                        help='prefix to save MLP predictions and targets')
    args = parser.parse_args()

    # prepare data
    X, y = load_mnist(mode='train', path='../data/')
    X /= 255.
    RNG(seed=42).shuffle(X)
    RNG(seed=42).shuffle(y)
    X_train = X[:args.n_train]
    y_train = y[:args.n_train]
    X_val = X[-args.n_val:]
    y_val = y[-args.n_val:]

    # train and save the RBM model
    if args.load:
        print "\nLoading model ...\n\n"
        rbm = BernoulliRBM.load_model(args.load)
    else:
        print "\nTraining model ...\n\n"
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

    # discriminative fine-tuning: initialize 2-layer MLP with
    # learned weights and train using backprop
    print "\nDiscriminative fine-tuning ...\n\n"

    # define and initialize MLP model
    first_layer_params = {}
    if not args.mlp_no_init:
        weights = rbm.get_tf_params(scope='weights')
        W = weights['W']
        hb = weights['hb']
        first_layer_params['weights'] = (W, hb)

    mlp = Sequential([
        Dense(args.n_hidden, input_shape=(784,),
              kernel_regularizer=regularizers.l2(args.mlp_l2),
              **first_layer_params),
        Activation('sigmoid'),
        Dense(10),
        Activation('softmax'),
    ])

    mlp.compile(optimizer=MultiAdam(lr=0.001,
                                    lr_multipliers={'dense_1': args.mlp_lrm[0],
                                                    'dense_2': args.mlp_lrm[1]}),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # load test data
    X_test, y_test = load_mnist(mode='test', path='../data/')
    X_test /= 255.

    # train and evaluate classifier
    with Stopwatch(verbose=True) as s:
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=2)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=2,
                                      patience=10, min_lr=1e-5)
        try:
            mlp.fit(X_train, one_hot(y_train, n_classes=10),
                    epochs=args.mlp_epochs, shuffle=False,
                    validation_data=(X_val, one_hot(y_val, n_classes=10)),
                    callbacks=[early_stopping, reduce_lr])
        except KeyboardInterrupt:
            pass

        y_pred = mlp.predict(X_test)
        y_pred = unhot(one_hot_decision_function(y_pred), n_classes=10)
        print "Test accuracy: {:.4f}".format(accuracy_score(y_test, y_pred))

    # save predictions, targets, and fine-tuned weights
    np.save(args.mlp_save_prefix + 'y_pred.npy', y_pred)
    np.save(args.mlp_save_prefix + 'y_test.npy', y_test)
    W_finetuned, _ = mlp.layers[0].get_weights()
    np.save(args.mlp_save_prefix + 'W_finetuned.npy', W_finetuned)


if __name__ == '__main__':
    main()
