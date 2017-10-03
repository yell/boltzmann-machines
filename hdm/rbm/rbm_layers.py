import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import Bernoulli, Multinomial, Normal

from hdm.base import SeedMixin


class BaseLayer(object):
    """Helper class that encapsulates one layer of stochastic units in RBM/DBM."""
    def __init__(self, n_units, tf_dtype=tf.float32):
        super(BaseLayer, self).__init__()
        self.n_units = n_units
        self.tf_dtype = tf_dtype

    def init(self, batch_size):
        """Randomly initialize states according to their distribution."""
        raise NotImplementedError('`init` is not implemented')

    def activation(self, x, b):
        """Compute activation of states according to the layer's distribution.

        Parameters
        ----------
        x : (n_units,) tf.Tensor
            Total input received (excluding bias).
        b : (n_units,) tf.Tensor
            Bias.
        """
        raise NotImplementedError('`activation` is not implemented')

    def _sample(self, means):
        """Sample states of the units by combining output from 2 previous functions."""
        raise NotImplementedError('`sample` is not implemented')

    def sample(self, means):
        T = self._sample(means).sample()
        return tf.cast(T, dtype=self.tf_dtype)


class BernoulliLayer(BaseLayer):
    def __init__(self, *args, **kwargs):
        super(BernoulliLayer, self).__init__(*args, **kwargs)

    def init(self, batch_size, random_seed=None):
        return tf.random_uniform([batch_size, self.n_units], minval=0., maxval=1.,
                                 dtype=self.tf_dtype, seed=random_seed, name='bernoulli_init')

    def activation(self, x, b):
        return tf.nn.sigmoid(x + b)

    def _sample(self, means):
        return Bernoulli(probs=means)


class MultinomialLayer(BaseLayer):
    def __init__(self, n_samples=100, sqrt_M=False, *args, **kwargs):
        super(MultinomialLayer, self).__init__(*args, **kwargs)
        self.n_samples = float(n_samples)
        self.sqrt_M = sqrt_M

    def init(self, batch_size, random_seed=None):
        t = tf.random_uniform([batch_size, self.n_units], minval=0., maxval=1.,
                              dtype=self.tf_dtype, seed=random_seed)
        t /= tf.reduce_sum(t)
        return tf.identity(t, name='multinomial_init')

    def activation(self, x, b):
        t = tf.nn.softmax(x + b)
        return np.sqrt(self.n_samples) * t if self.sqrt_M else self.n_samples * t

    def _sample(self, means):
        return Multinomial(total_count=self.n_samples, probs=tf.to_float(means / tf.reduce_sum(means)))

    def sample(self, means):
        T = super(MultinomialLayer, self).sample(means)
        if self.sqrt_M:
            T /= np.sqrt(self.n_samples)
        return T


class GaussianLayer(BaseLayer):
    def __init__(self, sigma, *args, **kwargs):
        super(GaussianLayer, self).__init__(*args, **kwargs)
        self.sigma = np.asarray(sigma)

    def init(self, batch_size, random_seed=None):
        t = tf.random_normal([batch_size, self.n_units],
                             dtype=self.tf_dtype, seed=random_seed)
        t = tf.multiply(t, self.sigma, name='gaussian_init')
        return t

    def activation(self, x, b):
        t = x * self.sigma + b
        return t

    def _sample(self, means):
        return Normal(loc=means, scale=tf.cast(self.sigma, dtype=self.tf_dtype))
