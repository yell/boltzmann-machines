import numpy as np
import tensorflow as tf

from base_rbm import BaseRBM


class BernoulliRBM(BaseRBM):
    """RBM with Bernoulli both visible and hidden units."""
    def __init__(self, model_path='b_rbm_model/',
                 **kwargs):
        super(BernoulliRBM, self).__init__(model_path=model_path, **kwargs)

    def _make_placeholders(self):
        super(BernoulliRBM, self)._make_placeholders_routine(h_rand_shape=[None, self.n_hidden])

    def _make_h_rand(self, X_batch):
        return self._rng.rand(X_batch.shape[0], self.n_hidden)

    def _make_v_rand(self, X_batch):
        return self._rng.rand(X_batch.shape[0], self.n_visible)

    def _free_energy(self, v):
        with tf.name_scope('free_energy'):
            tv = -tf.einsum('ij,j->i', v, self._vb)
            th = -tf.reduce_sum(tf.nn.softplus(self._propup(v)), axis=1)
            fe = tf.reduce_mean(tv + th, axis=0)
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
        super(MultinomialRBM, self)._make_placeholders_routine(h_rand_shape=[None, 1])

    def _make_h_rand(self, X_batch):
        return self._rng.rand(X_batch.shape[0], 1)

    def _make_v_rand(self, X_batch):
        return self._rng.rand(X_batch.shape[0], self.n_visible)

    def _h_means_given_v(self, v):
        with tf.name_scope('h_means_given_v'):
            h_means = tf.nn.softmax(self._propup(v))
        return h_means

    def _sample_h_given_v(self, h_means):
        with tf.name_scope('sample_h_given_v'):
            h_cumprobs = tf.cumsum(h_means, axis=-1)
            t = tf.to_int32(tf.greater_equal(h_cumprobs, self._h_rand))
            ind = tf.to_int32(tf.argmax(t, axis=-1))
            r = tf.to_int32(tf.range(tf.shape(ind)[0]))
            h_samples = tf.scatter_nd(tf.transpose([r, ind]),
                                      tf.ones_like(r),
                                      tf.to_int32(tf.shape(h_cumprobs)))
            h_samples = tf.to_float(h_samples)
        return h_samples

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
    def __init__(self,
                 learning_rate=1e-3,
                 sigma=1.,
                 model_path='g_rbm_model/', **kwargs):
        super(GaussianRBM, self).__init__(learning_rate=learning_rate,
                                          model_path=model_path, **kwargs)
        self.sigma = sigma
        if hasattr(self.sigma, '__iter__'):
            self._sigma_tmp = self.sigma = list(self.sigma)
        else:
            self._sigma_tmp = [self.sigma] * self.n_visible

    def _make_placeholders(self):
        super(GaussianRBM, self)._make_placeholders_routine(h_rand_shape=[None, self.n_hidden])
        with tf.name_scope('input_data'):
            # divide by resp. sigmas before any operation
            self._sigma = tf.Variable(tf.reshape(self._sigma_tmp, [1, self.n_visible]))
            self._X_batch = tf.divide(self._X_batch, self._sigma)

    def _make_h_rand(self, X_batch):
        return self._rng.rand(X_batch.shape[0], self.n_hidden)

    def _make_v_rand(self, X_batch):
        return self._rng.randn(X_batch.shape[0], self.n_visible)

    def _v_means_given_h(self, h):
        with tf.name_scope('v_means_given_h'):
            v_means = tf.matmul(a=h, b=self._W, transpose_b=True) * self._sigma
            v_means += self._vb
        # Need to multiply by 2 if used for pre-training as last layer of DBM,
        # but typically it is used as first layer where visible layers represent data
        return v_means

    def _sample_v_given_h(self, v_means):
        with tf.name_scope('sample_v_given_h'):
            v_samples = v_means + self._v_rand * self._sigma
        return v_samples

    def _free_energy(self, v):
        with tf.name_scope('free_energy'):
            t = tf.divide(tf.reshape(self._vb, [1, self.n_visible]), self._sigma)
            t2 = tf.square(tf.subtract(v, t))
            tv = 0.5 * tf.reduce_sum(t2, axis=1)
            th = -tf.reduce_sum(tf.nn.softplus(self._propup(v)), axis=1)
            fe = tf.reduce_mean(tv + th, axis=0)
        return fe


def bernoulli_vb_initializer(X):
    p = np.mean(X, axis=0)
    q = np.log(np.maximum(p, 1e-6) / np.maximum(1. - p, 1e-6))
    return q


if __name__ == '__main__':
    # run corresponding tests
    from utils.testing import run_tests
    from tests import test_rbm
    run_tests(__file__, test_rbm)
