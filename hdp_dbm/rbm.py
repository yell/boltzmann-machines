import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from base_rbm import BaseRBM
from utils.dataset import load_mnist


class BernoulliRBM(BaseRBM):
    """RBM with Bernoulli both visible and hidden units."""
    def __init__(self, model_path='b_rbm_model/',
                 **kwargs):
        super(BernoulliRBM, self).__init__(model_path=model_path, **kwargs)


class MultinomialRBM(BaseRBM):
    """RBM with Bernoulli visible and single Multinomial hidden unit.

    Parameters
    ----------
    n_hidden : int
        Number of possible states of a multinomial unit.
    """

    def __init__(self, model_path='m_rbm_model/',
                 **kwargs):
        super(MultinomialRBM, self).__init__(model_path=model_path, **kwargs)

    def _sample_h_given_v(self, v):
        with tf.name_scope('sample_h_given_v'):
            with tf.name_scope('h_probs'):
                h_probs = tf.nn.softmax(self._propup(v))
            with tf.name_scope('h_samples'):
                h_samples = tf.to_float(tf.less(self._h_rand, h_probs))
        return h_probs, h_samples


class GaussianRBM(BaseRBM):
    """RBM with Gaussian visible and Bernoulli hidden units."""
    pass


def plot_rbm_filters(W):
    plt.figure(figsize=(12, 12))
    for i in xrange(100):
        filters = W[:, i].reshape((28, 28))
        plt.subplot(10, 10, i + 1)
        plt.imshow(filters, cmap=plt.cm.gray, interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle('First 100 components extracted by RBM', fontsize=24)


def bernoulli_vb_initializer(X):
    p = np.mean(X, axis=0)
    q = np.log(np.maximum(p, 1e-15) / np.maximum(1. - p, 1e-15))
    return q


# if __name__ == '__main__':
#     # run corresponding tests
#     from utils.testing import run_tests
#     from tests import test_rbm
#     run_tests(__file__, test_rbm)


if __name__ == '__main__':
    # X, _ = load_mnist(mode='train', path='../data/')
    # X_val, _ = load_mnist(mode='test', path='../data/')
    # X = X[:1000]
    # X_val = X_val[:100]
    # X /= 255.
    # X_val /= 255.
    #
    # rbm = BernoulliRBM(n_visible=784,
    #                    n_hidden=256,
    #                    vb_init=bernoulli_vb_initializer(X),
    #                    n_gibbs_steps=1,
    #                    learning_rate=0.01,
    #                    momentum=0.9,#[0.5, 0.6, 0.7, 0.8, 0.9],
    #                    max_epoch=3,
    #                    batch_size=10,
    #                    L2=1e-4,
    #                    verbose=True,
    #                    random_seed=1337,
    #                    model_path='../models/rbm0/'
    #                    )
    # rbm.fit(X, X_val)
    # print rbm.get_weights()['W:0'][0][0]
    # rbm.set_params(max_epoch=10).fit(X, X_val)
    # print rbm.get_weights()['W:0'][0][0]
    # # print rbm.get_weights()['W:0'][0][0]
    # # rbm = BernoulliRBM.load_model('../models/b-rbm/').set_params(max_epoch=10).fit(X, X_val)
    # # print rbm.get_weights()['W:0'][0][0]
    #
    # rbm2 = BernoulliRBM(n_visible=784,
    #                    n_hidden=256,
    #                    vb_init=bernoulli_vb_initializer(X),
    #                    n_gibbs_steps=1,
    #                    learning_rate=0.01,
    #                    momentum=0.9,  # [0.5, 0.6, 0.7, 0.8, 0.9],
    #                    max_epoch=10,
    #                    batch_size=10,
    #                    L2=1e-4,
    #                    verbose=False,
    #                    random_seed=1337,
    #                    model_path='../models/b-rbm/'
    #                    )
    # rbm2.fit(X, X_val)
    # print rbm2.get_weights()['W:0'][0][0]
    # # plot_rbm_filters(rbm.get_weights()['W:0'])
    # # plt.show()
    import os
    from shutil import rmtree
    from numpy.testing import (assert_allclose,
                               assert_almost_equal)

    from hdp_dbm.utils import RNG
    from hdp_dbm.base_rbm import BaseRBM
    X = RNG(seed=1337).rand(30, 24)
    # X_val = RNG(seed=42).rand(16, 24)
    rbm_config = dict(n_visible=24,
                       n_hidden=16,
                       # verbose=True,
                       random_seed=1337,
                       compute_train_metrics_every_iter=10000,
                       compute_dfe_every_epoch=10000,
                       L2=0.)
    rbm = BaseRBM(max_epoch=3,
                  model_path='test_rbm_1/',
                  **rbm_config)
    # print rbm.get_params()
    # # assert_almost_equal(rbm.get_weights()['W:0'][0][0], -0.0094548017)
    # rbm.fit(X)
    # print rbm.get_weights()['W:0'][0][0]
    # print "#### [1]: 3 epochs done"
    # rbm_weights = rbm.set_params(max_epoch=10) \
    #     .fit(X) \
    #     .get_weights()
    # print "#### [1]: 10 epochs done"

    # # 2) train 3 (+save), load and train 7 more epochs
    # rbm2 = BaseRBM(max_epoch=3,
    #                model_path='test_rbm_2/',
    #                **rbm_config)
    # rbm2.fit(X)
    # print rbm2.get_weights()['W:0'][0][0]
    # print rbm.get_weights()['W:0'][0][0]
    # rbm2 = BaseRBM.load_model('test_rbm_2/')
        #      \
        # .set_params(max_epoch=10) \
        # .fit(X) \
        # .get_weights()
    # print rbm2.get_weights()['W:0'][0][0]
    # print "#### [2]: 10 epochs done"
    # assert_allclose(rbm_weights['W:0'], rbm2_weights['W:0'])
    # assert_allclose(rbm_weights['hb:0'], rbm2_weights['hb:0'])
    # assert_allclose(rbm_weights['vb:0'], rbm2_weights['vb:0'])
