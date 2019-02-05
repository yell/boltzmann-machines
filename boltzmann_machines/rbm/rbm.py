import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import Multinomial

from .env import *
from .base_rbm import BaseRBM
from layers import BernoulliLayer, MultinomialLayer, GaussianLayer


class BernoulliRBM(BaseRBM):
    """RBM with Bernoulli both visible and hidden units."""
    def __init__(self, model_path='b_rbm_model/', *args, **kwargs):
        super(BernoulliRBM, self).__init__(v_layer_cls=BernoulliLayer,
                                           h_layer_cls=BernoulliLayer,
                                           model_path=model_path, *args, **kwargs)

    def _free_energy(self, v):
        with tf.name_scope('free_energy'):
            T1 = -tf.einsum('ij,j->i', v, self._vb)
            T2 = -tf.reduce_sum(tf.nn.softplus(self._propup(v) + self._hb), axis=1)
            fe = tf.reduce_mean(T1 + T2, axis=0)
        return fe


class MultinomialRBM(BaseRBM):
    """RBM with Bernoulli visible and single Multinomial hidden unit
    (= multiple softmax units with tied weights).

    Parameters
    ----------
    n_hidden : int
        Number of possible states of a multinomial unit.
    n_samples : int
        Number of softmax units with shared weights
        (= number of samples from one softmax unit).

    References
    ----------
    [1] R. Salakhutdinov, A. Mnih, and G. Hinton. Restricted boltzmann
        machines for collaborative filtering, 2007.
    """
    def __init__(self, n_samples=100,
                 model_path='m_rbm_model/', *args, **kwargs):
        self.n_samples = n_samples
        super(MultinomialRBM, self).__init__(v_layer_cls=BernoulliLayer,
                                             h_layer_cls=MultinomialLayer,
                                             h_layer_params=dict(n_samples=self.n_samples),
                                             model_path=model_path, *args, **kwargs)

    def _free_energy(self, v):
        K = float(self.n_hidden)
        M = float(self.n_samples)
        with tf.name_scope('free_energy'):
            T1 = -tf.einsum('ij,j->i', v, self._vb)
            T2 = -tf.matmul(v, self._W)
            h_hat = Multinomial(total_count=M, logits=tf.ones([K])).sample()
            T3 = tf.einsum('ij,j->i', T2, h_hat)
            fe = tf.reduce_mean(T1 + T3, axis=0)
            fe += -tf.lgamma(M + K) + tf.lgamma(M + 1) + tf.lgamma(K)
        return fe

    def transform(self, *args, **kwargs):
        H = super(MultinomialRBM, self).transform(*args, **kwargs)
        H /= float(self.n_samples)
        return H


class GaussianRBM(BaseRBM):
    """RBM with Gaussian visible and Bernoulli hidden units.

    This implementation does not learn variances, but instead uses
    fixed, predetermined values. Input data should be pre-processed
    to have zero mean (or, equivalently, initialize visible biases
    to the negative mean of data). It can also be normalized to have
    unit variance. In the latter case use `sigma` equal to 1., as
    suggested in [1].

    Parameters
    ----------
    sigma : float, or iterable of such
        Standard deviations of visible units.

    References
    ----------
    [1] Hinton, G. "A Practical Guide to Training Restricted Boltzmann
        Machines" UTML TR 2010-003
    """
    def __init__(self, learning_rate=1e-3, sigma=1.,
                 model_path='g_rbm_model/', *args, **kwargs):
        self.sigma = sigma
        super(GaussianRBM, self).__init__(v_layer_cls=GaussianLayer,
                                          v_layer_params=dict(sigma=self.sigma),
                                          h_layer_cls=BernoulliLayer,
                                          learning_rate=learning_rate,
                                          model_path=model_path, *args, **kwargs)
        if hasattr(self.sigma, '__iter__'):
            self._sigma_tmp = self.sigma = np.asarray(self.sigma)
        else:
            self._sigma_tmp = np.repeat(self.sigma, self.n_visible)

    def _make_placeholders(self):
        super(GaussianRBM, self)._make_placeholders()
        with tf.name_scope('input_data'):
            # divide by resp. sigmas before any operation
            self._sigma = tf.Variable(self._sigma_tmp, dtype=self._tf_dtype, name='sigma')
            self._sigma = tf.reshape(self._sigma, [1, self.n_visible])
            self._X_batch = tf.divide(self._X_batch, self._sigma)

    def _free_energy(self, v):
        with tf.name_scope('free_energy'):
            T1 = tf.divide(tf.reshape(self._vb, [1, self.n_visible]), self._sigma)
            T2 = tf.square(tf.subtract(v, T1))
            T3 = 0.5 * tf.reduce_sum(T2, axis=1)
            T4 = -tf.reduce_sum(tf.nn.softplus(self._propup(v) + self._hb), axis=1)
            fe = tf.reduce_mean(T3 + T4, axis=0)
        return fe


def logit_mean(X):
    p = np.mean(X, axis=0)
    p = np.clip(p, 1e-7, 1. - 1e-7)
    q = np.log(p / (1. - p))
    return q


if __name__ == '__main__':
    # run corresponding tests
    from utils.testing import run_tests
    from tests import test_rbm as t
    run_tests(__file__, t)
