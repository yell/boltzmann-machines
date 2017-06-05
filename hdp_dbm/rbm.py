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

    def _make_placeholders(self):
        super(BernoulliRBM, self)._make_placeholders_routine(h_rand_samples=self.n_hidden)

    def _make_tf_feed_dict(self, *args, **kwargs):
        return super(BernoulliRBM, self)._make_tf_feed_dict_routine(self.n_hidden,
                                                                    *args,
                                                                    **kwargs)

    def _free_energy(self, v):
        with tf.name_scope('free_energy'):
            fe = -tf.einsum('ij,j->i', v, self._vb)
            fe -= tf.reduce_sum(tf.nn.softplus(self._propup(v)), axis=1)
            fe = tf.reduce_mean(fe, axis=0)
        return fe


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

    def _make_placeholders(self):
        super(MultinomialRBM, self)._make_placeholders_routine(h_rand_samples=1)

    def _make_tf_feed_dict(self, *args, **kwargs):
        return super(MultinomialRBM, self)._make_tf_feed_dict_routine(1, *args, **kwargs)

    def _sample_h_given_v(self, v):
        with tf.name_scope('sample_h_given_v'):
            with tf.name_scope('h_probs'):
                h_probs = tf.nn.softmax(self._propup(v))
            with tf.name_scope('h_samples'):
                h_cumprobs = tf.cumsum(h_probs, axis=-1)
                m = tf.to_int32(tf.greater_equal(h_cumprobs, self._h_rand))
                ind = tf.to_int32(tf.argmax(m, axis=-1))
                z = tf.to_int32(tf.range(tf.shape(ind)[0]))
                h_samples = tf.scatter_nd(tf.transpose([z, ind]),
                                          tf.ones_like(z),
                                          tf.to_int32(tf.shape(h_cumprobs)))
                h_samples = tf.to_float(h_samples)
        return h_probs, h_samples

    def _free_energy(self, v):
        with tf.name_scope('free_energy'):
            fe = -tf.einsum('ij,j->i', v, self._vb)
            fe -= tf.reduce_sum(tf.matmul(v, self._W), axis=1)
            fe = tf.reduce_mean(fe, axis=0) - tf.reduce_sum(self._hb)
        return fe


class GaussianRBM(BaseRBM):
    """RBM with Gaussian visible and Bernoulli hidden units."""
    pass


def bernoulli_vb_initializer(X):
    p = np.mean(X, axis=0)
    q = np.log(np.maximum(p, 1e-15) / np.maximum(1. - p, 1e-15))
    return q


def plot_rbm_filters(W):
    plt.figure(figsize=(12, 12))
    for i in xrange(10):
        filters = W[:, i].reshape((28, 28))
        plt.subplot(10, 10, i + 1)
        plt.imshow(filters, cmap=plt.cm.gray, interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle('First 100 components extracted by RBM', fontsize=24)





# if __name__ == '__main__':
#     # run corresponding tests
#     from utils.testing import run_tests
#     from tests import test_rbm
#     run_tests(__file__, test_rbm)


if __name__ == '__main__':
    X, _ = load_mnist(mode='train', path='../data/')
    X_val, _ = load_mnist(mode='test', path='../data/')
    X = X[:10000]
    X_val = X_val[:1000]
    X /= 255.
    X_val /= 255.

    rbm = MultinomialRBM(n_visible=784,
                       n_hidden=10,
                       vb_init=bernoulli_vb_initializer(X),
                       n_gibbs_steps=1,
                       learning_rate=0.01,
                       momentum=[0.5, 0.6, 0.7, 0.8, 0.9],
                       max_epoch=10,
                       batch_size=10,
                       L2=1e-4,
                       verbose=True,
                       random_seed=1337,
                       model_path='../models/m-rbm2/')
    rbm.fit(X, X_val)

    # rbm = MultinomialRBM.load_model('../models/m-rbm/')
    plot_rbm_filters(rbm.get_weights()['W:0'])
    plt.show()
