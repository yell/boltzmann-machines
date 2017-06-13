import numpy as np
import tensorflow as tf


class BaseLayer(object):
    """Helper class that encapsulates one layer of stochastic units in RBM/DBM."""
    def __init__(self, n_units, tf_dtype=tf.float32):
        self.n_units = n_units
        self.tf_dtype = tf_dtype

    def init(self, batch_size, random_seed=None):
        """Randomly initialize states according to their distribution."""
        raise NotImplementedError('`init` is not implemented')

    def activation(self, x, b):
        """Compute activation of states according to the distribution.

        Parameters
        ----------
        x - total input received (incl. bias)
        b - bias
        """
        raise NotImplementedError('`activation` is not implemented')

    def get_rand_shape(self):
        """Return shape of respective placeholder."""
        return [None, self.n_units]

    def make_rand(self, batch_size, rng):
        """Generate random data that will be passed to feed_dict."""
        raise NotImplementedError('`make_rand` is not implemented')

    def sample(self, rand_data, means):
        """Sample states of the units by combining output from 2 previous functions."""
        raise NotImplementedError('`sample` is not implemented')


class BernoulliLayer(BaseLayer):
    def __init__(self, **kwargs):
        super(BernoulliLayer, self).__init__(**kwargs)

    def init(self, batch_size, random_seed=None):
        return tf.random_uniform([batch_size, self.n_units], minval=0., maxval=1.,
                                 dtype=self.tf_dtype, seed=random_seed, name='bernoulli_init')

    def activation(self, x, b):
        return tf.nn.sigmoid(x + b)

    def make_rand(self, batch_size, rng):
        return rng.rand(batch_size, self.n_units)

    def sample(self, rand_data, means):
        return tf.cast(tf.less(rand_data, means), dtype=self.tf_dtype)


class MultinomialLayer(BaseLayer):
    def __init__(self, **kwargs):
        super(MultinomialLayer, self).__init__(**kwargs)

    def init(self, batch_size, random_seed=None):
        t = tf.random_uniform([batch_size, self.n_units], minval=0., maxval=1.,
                              dtype=self.tf_dtype, seed=random_seed)
        t /= tf.reduce_sum(t)
        return tf.identity(t, name='multinomial_init')

    def activation(self, x, b):
        return tf.nn.softmax(x + b)

    def get_rand_shape(self):
        return [None, 1]

    def make_rand(self, batch_size, rng):
        return rng.rand(batch_size, 1)

    def sample(self, rand_data, means):
        cumprobs = tf.cumsum(means, axis=-1)
        t = tf.to_int32(tf.greater_equal(cumprobs, rand_data))
        ind = tf.to_int32(tf.argmax(t, axis=-1))
        r = tf.to_int32(tf.range(tf.shape(ind)[0]))
        samples = tf.scatter_nd(tf.transpose([r, ind]),
                                tf.ones_like(r),
                                tf.to_int32(tf.shape(cumprobs)))
        return tf.cast(samples, dtype=self.tf_dtype)


class GaussianLayer(BaseLayer):
    def __init__(self, sigma, **kwargs):
        super(GaussianLayer, self).__init__(**kwargs)
        self.sigma = np.asarray(sigma)

    def init(self, batch_size, random_seed=None):
        t = tf.random_normal([batch_size, self.n_units], dtype=self.tf_dtype, seed=random_seed)
        t = tf.multiply(t, self.sigma, name='gaussian_init')
        return t

    def activation(self, x, b):
        t = x * self.sigma + b
        return t

    def make_rand(self, batch_size, rng):
        return rng.randn(batch_size, self.n_units)

    def sample(self, rand_data, means):
        return rand_data * self.sigma + means
