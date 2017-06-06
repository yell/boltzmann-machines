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

    def _sample_h_given_v(self, v):
        with tf.name_scope('sample_h_given_v'):
            with tf.name_scope('h_probs'):
                h_means = tf.nn.softmax(self._propup(v))
            with tf.name_scope('h_samples'):
                h_cumprobs = tf.cumsum(h_means, axis=-1)
                t = tf.to_int32(tf.greater_equal(h_cumprobs, self._h_rand))
                ind = tf.to_int32(tf.argmax(t, axis=-1))
                r = tf.to_int32(tf.range(tf.shape(ind)[0]))
                h_samples = tf.scatter_nd(tf.transpose([r, ind]),
                                          tf.ones_like(r),
                                          tf.to_int32(tf.shape(h_cumprobs)))
                h_samples = tf.to_float(h_samples)
        return h_means, h_samples

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
    to have zero mean and unit variance, as suggested in [2].

    Parameters
    ----------
    sigma : float, or iterable of such
        Standard deviations of visible units.
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

    def _sample_v_given_h(self, h):
        with tf.name_scope('sample_v_given_h'):
            with tf.name_scope('v_means'):
                v_means = tf.nn.sigmoid(self._propdown(h))
            with tf.name_scope('v_samples'):
                v_samples = tf.matmul(a=h, b=self._W, transpose_b=True) * self._sigma
                v_samples += self._vb
                v_samples += self._v_rand * self._sigma
        return v_means, v_samples

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
    q = np.log(np.maximum(p, 1e-15) / np.maximum(1. - p, 1e-15))
    return q


def plot_rbm_filters(W):
    plt.figure(figsize=(12, 12))
    for i in xrange(100):
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
    X, y = load_mnist(mode='train', path='../data/')
    X_val, _ = load_mnist(mode='test', path='../data/')
    X /= 255.
    X_val /= 255.
    # X_new = np.zeros_like(X)
    # from utils import make_k_folds
    # start = 0
    # for i_b in make_k_folds(y, n_folds=6000, shuffle=True, stratify=True, random_seed=1337):
    #     X_new[start:(start + len(i_b))] = X[i_b]
    #     start += len(i_b)
    # X = X_new
    #
    # for lr in (0.01, 0.001):
    #     for L2 in (0., 1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
    #         rbm = BernoulliRBM(n_visible=784,
    #                            n_hidden=1024,
    #                            vb_init=bernoulli_vb_initializer(X),
    #                            n_gibbs_steps=1,
    #                            learning_rate=lr,
    #                            momentum=[0.5] * 20 + [0.6, 0.7, 0.8, 0.9],
    #                            max_epoch=40,
    #                            batch_size=10,
    #                            L2=L2,
    #                            verbose=True,
    #                            random_seed=1337,
    #                            model_path='../models/L2-{0}-lr-{1}/'.format(L2, lr))
    #         rbm.fit(X, X_val)

    X = X[:1000]
    X_val = X_val[:100]
    rbm = GaussianRBM(n_visible=784,
                      n_hidden=256,
                      vb_init=0.,# bernoulli_vb_initializer(X),
                      n_gibbs_steps=1,
                      learning_rate=0.0001,
                      momentum=[0.5] * 5 * 1000 + [0.9],
                      max_epoch=10,
                      batch_size=10,
                      L2=1e-5,
                      metrics_config=dict(
                          l2_loss=True,
                          msre=True,
                          pll=True,
                          dfe=True,
                          dfe_fmt='.3f',
                          train_metrics_every_iter=10,
                      ),
                      verbose=True,
                      random_seed=1337,
                      model_path='../models/g-rbm/')
    rbm.fit(X, X_val)
    rbm = GaussianRBM.load_model('../models/g-rbm/')
    rbm.set_params(max_epoch=24, batch_size=7)
    rbm.fit(X, X_val)
    # print rbm.get_weights()['W:0'][0][0]
    # plot_rbm_filters(rbm.get_weights()['W:0'])
    # plt.show()
