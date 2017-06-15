import numpy as np
import tensorflow as tf

from base_rbm import BaseRBM
from layers import BernoulliLayer, MultinomialLayer, GaussianLayer


class BernoulliRBM(BaseRBM):
    """RBM with Bernoulli both visible and hidden units."""
    def __init__(self, model_path='b_rbm_model/', *args, **kwargs):
        super(BernoulliRBM, self).__init__(model_path=model_path, *args, **kwargs)
        self._v_layer = BernoulliLayer(n_units=self.n_visible,
                                       tf_dtype=self._tf_dtype,
                                       random_seed=self.make_random_seed())
        self._h_layer = BernoulliLayer(n_units=self.n_hidden,
                                       tf_dtype=self._tf_dtype,
                                       random_seed=self.make_random_seed())

    def _free_energy(self, v):
        with tf.name_scope('free_energy'):
            tv = -tf.einsum('ij,j->i', v, self._vb)
            th = -tf.reduce_sum(tf.nn.softplus(self._propup(v) + self._hb), axis=1)
            fe = tf.reduce_mean(tv + th, axis=0)
        return fe


class MultinomialRBM(BaseRBM):
    """RBM with Bernoulli visible and single Multinomial hidden unit
    (= multiple softmax units with tied weights).

    Parameters
    ----------
    n_hidden : int
        Number of possible states of a multinomial unit.
    """
    def __init__(self, model_path='m_rbm_model/', *args, **kwargs):
        super(MultinomialRBM, self).__init__(model_path=model_path, *args, **kwargs)
        self._v_layer = BernoulliLayer(n_units=self.n_visible,
                                       tf_dtype=self._tf_dtype,
                                       random_seed=self.make_random_seed())
        self._h_layer = MultinomialLayer(n_units=self.n_hidden,
                                         tf_dtype=self._tf_dtype,
                                         random_seed=self.make_random_seed())

    def _free_energy(self, v):
        with tf.name_scope('free_energy'):
            tv = -tf.einsum('ij,j->i', v, self._vb)
            th = -tf.reduce_sum(tf.matmul(v, self._W), axis=1)
            fe = tf.reduce_mean(tv + th, axis=0) - tf.reduce_sum(self._hb)
        return fe


class GaussianRBM(BaseRBM):
    """RBM with Gaussian visible and Bernoulli hidden units.

    This implementation does not learn variances, but instead uses
    fixed, predetermined values. Input data should be pre-processed
    to have zero mean and unit variance, as suggested in [1].

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
        super(GaussianRBM, self).__init__(learning_rate=learning_rate,
                                          model_path=model_path, *args, **kwargs)
        self.sigma = sigma
        if hasattr(self.sigma, '__iter__'):
            self._sigma_tmp = self.sigma = list(self.sigma)
        else:
            self._sigma_tmp = [self.sigma] * self.n_visible

        self._v_layer = GaussianLayer(sigma=self.sigma,
                                      n_units=self.n_visible,
                                      tf_dtype=self._tf_dtype,
                                      random_seed=self.make_random_seed())
        self._h_layer = BernoulliLayer(n_units=self.n_hidden,
                                       tf_dtype=self._tf_dtype,
                                       random_seed=self.make_random_seed())

    def _make_placeholders(self):
        super(GaussianRBM, self)._make_placeholders()
        with tf.name_scope('input_data'):
            # divide by resp. sigmas before any operation
            self._sigma = tf.Variable(self._sigma_tmp, dtype=self._tf_dtype, name='sigma')
            self._sigma = tf.reshape(self._sigma, [1, self.n_visible])
            self._X_batch = tf.divide(self._X_batch, self._sigma)

    def _free_energy(self, v):
        with tf.name_scope('free_energy'):
            t = tf.divide(tf.reshape(self._vb, [1, self.n_visible]), self._sigma)
            t2 = tf.square(tf.subtract(v, t))
            tv = 0.5 * tf.reduce_sum(t2, axis=1)
            th = -tf.reduce_sum(tf.nn.softplus(self._propup(v) + self._hb), axis=1)
            fe = tf.reduce_mean(tv + th, axis=0)
        return fe


def init_sigmoid_vb_from_data(X):
    p = np.mean(X, axis=0)
    p = np.clip(p, 1e-7, 1e-7)
    q = np.log(p / (1. - p))
    return q


if __name__ == '__main__':
    # run corresponding tests
    from utils.testing import run_tests
    from tests import test_rbm
    run_tests(__file__, test_rbm)
