import tensorflow as tf
from matplotlib import pyplot as plt

from base import TensorFlowModel
from utils import batch_iter
from utils.dataset import load_mnist


class BaseRBM(TensorFlowModel):
    def __init__(self, n_visible=784, n_hidden=256, n_gibbs_steps=1,
                 learning_rate=0.1, momentum=0.9, batch_size=10, max_epoch=10,
                 verbose=False, shuffle=True,
                 model_dirpath='rbm_model/', **kwargs):
        super(BaseRBM, self).__init__(model_dirpath=model_dirpath, **kwargs)
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_gibbs_steps = n_gibbs_steps

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size
        self.max_epoch = max_epoch

        self.verbose = verbose
        self.shuffle = shuffle

        self.epoch = 0

        # placeholders
        self._input_batch = None
        self._h_samples = None
        self._v_samples = None

        # weights and other variables
        self.W_ = None
        self.hb_ = None
        self.vb_ = None
        self._dW = None
        self._dhb = None
        self._dvb = None

        # operations
        self._W_update = None
        self._hb_update = None
        self._vb_update = None
        self._loss = None

    def _sample_h_given_v(self, v):
        """Sample from P(h|v)."""
        h_means = tf.nn.sigmoid(tf.matmul(v, self.W_) + self.hb_)
        h_samples = tf.to_float(tf.less(self._h_samples, h_means))
        return h_means, h_samples

    def _sample_v_given_h(self, h):
        """Sample from P(v|h)."""
        v_means = tf.nn.sigmoid(tf.matmul(h, tf.transpose(self.W_)) + self.vb_)
        v_samples = tf.to_float(tf.less(self._v_samples, v_means))
        return v_means, v_samples

    def _make_tf_model(self):
        # create placeholders
        self._input_batch = tf.placeholder('float', [None, self.n_visible], name='input_batch')
        self._h_samples = tf.placeholder('float', [None, self.n_hidden], name='h_samples')
        self._v_samples = tf.placeholder('float', [None, self.n_visible], name='v_samples')

        # create variables
        W_tensor = tf.random_normal((self.n_visible, self.n_hidden),
                                    mean=0.0, stddev=0.01, seed=self.random_seed)
        self.W_  = tf.Variable(W_tensor, name='W', dtype=tf.float32)
        self.hb_  = tf.Variable(tf.zeros((self.n_hidden,)), name='hb', dtype=tf.float32)
        self.vb_ = tf.Variable(tf.zeros((self.n_visible,)), name='vb', dtype=tf.float32)

        self._dW = tf.Variable(tf.zeros((self.n_visible, self.n_hidden)), name='dW', dtype=tf.float32)
        self._dhb = tf.Variable(tf.zeros((self.n_hidden,)), name='dhb', dtype=tf.float32)
        self._dvb = tf.Variable(tf.zeros((self.n_visible,)), name='dvb', dtype=tf.float32)

        # run Gibbs chain
        h0_means, h0_samples = self._sample_h_given_v(self._input_batch)
        h_means, v_means, v_samples = None, None, None
        h_samples = h0_samples
        for _ in xrange(self.n_gibbs_steps):
            v_means, v_samples = self._sample_v_given_h(h_samples)
            h_means, h_samples = self._sample_h_given_v(v_samples)

        # update parameters
        dW_positive = tf.matmul(tf.transpose(self._input_batch), h0_means)
        dW_negative = tf.matmul(tf.transpose(v_samples), h_means)
        self._dW = self.momentum * self._dW + (dW_positive - dW_negative)
        self._W_update = self.W_.assign_add(self.learning_rate * self._dW)

        self._dhb = self.momentum * self._dhb + tf.reduce_mean(h0_means - h_means, axis=0)
        self._hb_update = self.hb_.assign_add(self.learning_rate * self._dhb)

        self._dvb = self.momentum * self._dvb + tf.reduce_mean(self._input_batch - v_samples, axis=0)
        self._vb_update = self.vb_.assign_add(self.learning_rate * self._dvb)

        # collect summary
        self._loss = tf.sqrt(tf.reduce_mean(tf.square(self._input_batch - v_means)))
        tf.summary.scalar("loss", self._loss)

    def _make_tf_feed_dict(self, X_batch):
        return {
            'input_batch:0': X_batch,
            'h_samples:0': self._rng.rand(X_batch.shape[0], self.n_hidden),
            'v_samples:0': self._rng.rand(X_batch.shape[0], self.n_visible)
        }

    def _train_epoch(self, X, X_val):
        self.PW_ = tf.Print(self.W_, [self.W_], message="W: ")
        updates = (self._W_update, self._hb_update, self._vb_update, self.PW_)
        for X_batch in batch_iter(X, batch_size=self.batch_size):
            self._tf_session.run(updates, feed_dict=self._make_tf_feed_dict(X_batch=X_batch))

        if X_val is not None:
            summary_str, loss = self._tf_session.run((self._tf_merged_summaries, self._loss),
                                                     feed_dict=self._make_tf_feed_dict(X_val))
            if self.save_model:
                self._tf_summary_writer.add_summary(summary_str, self.epoch)
            if self.verbose:
                s = "epoch: {0:{1}}/{2} - loss: {3:.4f}"
                print s.format(self.epoch, len(str(self.max_epoch)), self.max_epoch, loss)
        else:
            print "epoch {0}".format(self.epoch)

    def _fit(self, X, X_val=None, *args, **kwargs):
        if self.shuffle:
            self._rng.shuffle(X)
        while self.epoch < self.max_epoch:
            self.epoch += 1
            self._train_epoch(X, X_val)



class BernoulliRBM(BaseRBM):
    """Bernoulli-Bernoulli RBM."""
    pass


class GaussianRBM(BaseRBM):
    """Gaussian-Bernoulli RBM."""
    pass


class MultinomialRBM(BaseRBM):
    """Bernoulli-Multinomial RBM."""
    pass


def plot_rbm_filters(W):
    plt.figure(figsize=(12, 12))
    for i in xrange(64):
        filters = W[:, i].reshape((28, 28))
        plt.subplot(8, 8, i + 1)
        plt.imshow(filters, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle('First 64 components extracted by RBM', fontsize=24)


if __name__ == '__main__':
    # rbm = BaseRBM(model_dirpath='../models/rbm1')
    # print rbm.get_params()

    # X, _ = load_mnist(mode='train', path='../data/')
    # X_val, _ = load_mnist(mode='test', path='../data/')
    # X = X[:2000]
    # X_val = X_val[:100]
    # X /= 255.
    # X_val /= 255.
    #
    # rbm = BaseRBM(n_visible=784,
    #               n_hidden=128,
    #               n_gibbs_steps=1,
    #               learning_rate=0.001,
    #               momentum=0.9,
    #               batch_size=10,
    #               max_epoch=3,
    #               verbose=True,
    #               random_seed=1337,
    #               model_dirpath='../models/rbm_3')
    # rbm.fit(X, X_val)
    #
    #
    # rbm = BaseRBM.load_model('../models/rbm_1')
    # rbm.set_params(max_epoch=16)
    # # print rbm.get_weights()
    # rbm.fit(X, X_val)

    # weights = rbm.get_weights()
    # W = weights['W_']
    # plot_rbm_filters(W)
    # plt.show()

    from utils import RNG
    X = RNG(seed=1337).rand(32, 256)
    rbm = BaseRBM(n_visible=256, n_hidden=100, max_epoch=3,
                  verbose=True, shuffle=False, model_dirpath='test_rbm_1',
                  random_seed=1337)
    rbm.fit(X)
    print rbm.get_weights()['W_'][0][0]
    rbm2 = BaseRBM.load_model('test_rbm_1')
    rbm2.set_params(max_epoch=10) # 7 more iterations
    rbm2.fit(X)
    rbm2_weights = rbm2.get_weights()
    rbm3 = BaseRBM(n_visible=256, n_hidden=100, max_epoch=10,
                   verbose=True, shuffle=False, model_dirpath='test_rbm_2',
                   random_seed=1337)
    rbm3.fit(X)
    rbm3_weights = rbm3.get_weights()
    import numpy as np
    np.testing.assert_allclose(rbm2_weights['W_'], rbm3_weights['W_'])
