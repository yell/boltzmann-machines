import numpy as np
import tensorflow as tf
from tensorflow.core.framework import summary_pb2

from base import TensorFlowModel, run_in_tf_session
from utils import batch_iter, tbatch_iter, make_inf_generator


class BaseRBM(TensorFlowModel):
    """
    Parameters
    ----------
    learning_rate, momentum : float, iterable, or generator
    vb_init : float or iterable

    References
    ----------
    [1] Goodfellow I. et. al. "Deep Learning".
    [2] Hinton, G. "A Practical Guide to Training Restricted Boltzmann
        Machines" UTML TR 2010-003
    [3] Restricted Boltzmann Machines (RBMs), Deep Learning Tutorial
        url: http://deeplearning.net/tutorial/rbm.html
    """
    def __init__(self, n_visible=784, n_hidden=256, n_gibbs_steps=1,
                 w_std=0.01, hb_init=0., vb_init=0.,
                 learning_rate=0.1, momentum=0.9, max_epoch=10, batch_size=10, L2=1e-4,
                 compute_train_metrics_every_iter=10, compute_val_metrics_every_epoch=1,
                 compute_dfe_every_epoch=2, n_batches_for_dfe=10,
                 verbose=False, model_path='rbm_model/', **kwargs):
        super(BaseRBM, self).__init__(model_path=model_path, **kwargs)
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_gibbs_steps = n_gibbs_steps

        self.w_std = w_std
        self.hb_init = hb_init

        # visible biases can be initialized with list of values,
        # because it is often helpful to initialize i-th visible bias
        # with value log(p_i / (1 - p_i)), p_i = fraction of training
        # vectors where i-th unit is on, as proposed in [2]
        self.vb_init = vb_init
        if hasattr(self.vb_init, '__iter__'):
            self._vb_init = self.vb_init = list(self.vb_init)
        else:
            self._vb_init = [self.vb_init] * self.n_visible

        self.learning_rate = learning_rate
        self._learning_rate_gen = None
        self.momentum = momentum
        self._momentum_gen = None
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.L2 = L2

        self.compute_train_metrics_every_iter = compute_train_metrics_every_iter
        self.compute_val_metrics_every_epoch = compute_val_metrics_every_epoch
        self.compute_dfe_every_epoch = compute_dfe_every_epoch
        self.n_batches_for_dfe = n_batches_for_dfe
        self.verbose = verbose

        # current epoch and iteration
        self.epoch = 0
        self.iter = 0

        # input data
        self._X_batch = None
        self._h_rand = None
        self._v_rand = None
        self._pll_rand = None
        self._learning_rate = None
        self._momentum = None

        # weights
        self._W = None
        self._hb = None
        self._vb = None

        # grads
        self._dW = None
        self._dhb = None
        self._dvb = None

        # operations
        self._train_op = None
        self._transform_op = None
        self._msre = None
        self._pseudo_loglik = None
        self._free_energy_op = None

    def _make_init_op(self):
        # create placeholders (input data)
        with tf.name_scope('input_data'):
            self._X_batch = tf.placeholder(tf.float32, [None, self.n_visible], name='X_batch')
            self._h_rand = tf.placeholder(tf.float32, [None, self.n_hidden], name='h_rand')
            self._v_rand = tf.placeholder(tf.float32, [None, self.n_visible], name='v_rand')
            self._pll_rand = tf.placeholder(tf.int32, [None], name='pll_rand')
            self._learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
            self._momentum = tf.placeholder(tf.float32, [], name='momentum')

        # create variables (weights and grads)
        with tf.name_scope('weights'):
            W_tensor = tf.random_normal((self.n_visible, self.n_hidden),
                                        mean=0.0, stddev=self.w_std, seed=self.random_seed)
            self._W = tf.Variable(W_tensor, name='W', dtype=tf.float32)
            self._hb = tf.Variable(self.hb_init * tf.ones((self.n_hidden,)), name='hb', dtype=tf.float32)
            self._vb = tf.Variable(self._vb_init, name='vb', dtype=tf.float32)
            tf.summary.histogram('W', self._W)
            tf.summary.histogram('hb', self._hb)
            tf.summary.histogram('vb', self._vb)

        with tf.name_scope('grads'):
            self._dW = tf.Variable(tf.zeros((self.n_visible, self.n_hidden)), name='dW', dtype=tf.float32)
            self._dhb = tf.Variable(tf.zeros((self.n_hidden,)), name='dhb', dtype=tf.float32)
            self._dvb = tf.Variable(tf.zeros((self.n_visible,)), name='dvb', dtype=tf.float32)
            tf.summary.histogram('dW', self._dW)
            tf.summary.histogram('dhb', self._dhb)
            tf.summary.histogram('dvb', self._dvb)

    def _propup(self, v):
        with tf.name_scope('prop_up'):
            h = tf.matmul(v, self._W) + self._hb
        return h

    def _propdown(self, h):
        with tf.name_scope('prop_down'):
            v = tf.matmul(a=h, b=self._W, transpose_b=True) + self._vb
        return v

    def _sample_h_given_v(self, v):
        """Sample from P(h|v)."""
        with tf.name_scope('sample_h_given_v'):
            with tf.name_scope('h_probs'):
                h_probs = tf.nn.sigmoid(self._propup(v))
            with tf.name_scope('h_samples'):
                h_samples = tf.to_float(tf.less(self._h_rand, h_probs))
        return h_probs, h_samples

    def _sample_v_given_h(self, h):
        """Sample from P(v|h)."""
        with tf.name_scope('sample_v_given_h'):
            with tf.name_scope('v_probs'):
                v_probs = tf.nn.sigmoid(self._propdown(h))
            with tf.name_scope('v_samples'):
                v_samples = tf.to_float(tf.less(self._v_rand, v_probs))
        return v_probs, v_samples

    def _free_energy(self, v):
        """Compute free energy of a visible vectors `v`."""
        with tf.name_scope('free_energy'):
            fe = -tf.einsum('ij,j->i', v, self._vb)
            fe -= tf.reduce_sum(tf.nn.softplus(self._propup(v)), axis=1)
            fe = tf.reduce_mean(fe, axis=0)
        return fe

    def _make_train_op(self):
        # Run Gibbs chain for specified number of steps.
        # According to [2], the training goes less noisy and slightly faster, if
        # sampling used for states of hidden units driven by the data, and probabilities
        # for ones driven by reconstructions, and if probabilities used for visible units,
        # both driven by data and by reconstructions.
        with tf.name_scope('gibbs_chain'):
            h0_means, h0_samples = self._sample_h_given_v(self._X_batch)
            v_probs, v_samples = None, None
            h_probs, h_samples = None, None
            h_states, v_states = h0_samples, None
            for _ in xrange(self.n_gibbs_steps):
                with tf.name_scope('sweep'):
                    v_probs, v_samples = self._sample_v_given_h(h_states)
                    v_states = v_probs
                    h_probs, h_samples = self._sample_h_given_v(v_states)
                    h_states = h_probs

        # encoded data, used by the transform method
        with tf.name_scope('transform_op'):
            transform_op = tf.identity(h_probs)
            tf.add_to_collection('transform_op', transform_op)

        # compute gradients estimates (= positive - negative associations)
        with tf.name_scope('grads_estimates'):
            N = tf.cast(tf.shape(self._X_batch)[0], dtype=tf.float32)
            with tf.name_scope('dW'):
                dW_positive = tf.matmul(self._X_batch, h0_means, transpose_a=True)
                dW_negative = tf.matmul(v_samples, h_probs, transpose_a=True)
                dW = (dW_positive - dW_negative) / N - self.L2 * self._W
            with tf.name_scope('dhb'):
                dhb = tf.reduce_mean(h0_means - h_probs, axis=0) # == sum / N
            with tf.name_scope('dvb'):
                dvb = tf.reduce_mean(self._X_batch - v_samples, axis=0) # == sum / N

        # update parameters
        with tf.name_scope('momentum_updates'):
            with tf.name_scope('dW'):
                dW_update = self._dW.assign(self._learning_rate * (self._momentum * self._dW + dW))
                W_update = self._W.assign_add(dW_update)
            with tf.name_scope('dhb'):
                dhb_update = self._dhb.assign(self._learning_rate * (self._momentum * self._dhb + dhb))
                hb_update = self._hb.assign_add(dhb_update)
            with tf.name_scope('dvb'):
                dvb_update = self._dvb.assign(self._learning_rate * (self._momentum * self._dvb + dvb))
                vb_update = self._vb.assign_add(dvb_update)

        # assemble train_op
        with tf.name_scope('train_op'):
            train_op = tf.group(W_update, hb_update, vb_update)
            tf.add_to_collection('train_op', train_op)

        # compute metrics
        with tf.name_scope('mean_squared_recon_error'):
            msre = tf.reduce_mean(tf.square(self._X_batch - v_probs))
            tf.add_to_collection('msre', msre)

        # Since reconstruction error is fairly poor measure of performance,
        # as this is not what CD-k learning algorithm aims to minimize [2],
        # compute (per sample average) pseudo-loglikelihood (proxy to likelihood)
        # instead, which not only is much more cheaper to compute, but also is
        # an asymptotically consistent estimate of the true log-likelihood [1].
        # More specifically, PLL computed using approximation as in [3].
        with tf.name_scope('pseudo_loglikelihood'):
            x = self._X_batch
            # randomly corrupt one feature in each sample
            x_ = tf.identity(x)
            ind = tf.transpose([tf.range(tf.shape(x)[0]), self._pll_rand])
            m = tf.SparseTensor(indices=tf.cast(ind, tf.int64),
                                values=tf.cast(tf.ones_like(self._pll_rand), tf.float32),
                                dense_shape=tf.cast(tf.shape(x_), tf.int64))
            x_ = tf.multiply(x_, -tf.sparse_tensor_to_dense(m, default_value=-1))
            x_ = tf.sparse_add(x_, m)

            # TODO: should change to tf.log_sigmoid when updated to r1.2
            pseudo_loglik = -tf.constant(self.n_visible, dtype='float') *\
                             tf.nn.softplus(-(self._free_energy(x_) -
                                              self._free_energy(x)))
            tf.add_to_collection('pseudo_loglik', pseudo_loglik)

        # add also free energy of input batch to collection (for dfe)
        free_energy_op = self._free_energy(self._X_batch)
        tf.add_to_collection('free_energy_op', free_energy_op)

        # collect summaries
        tf.summary.scalar('msre', msre)
        tf.summary.scalar('pseudo_loglik', pseudo_loglik)

    def _make_tf_model(self):
        self._make_init_op()
        self._make_train_op()

    def _make_tf_feed_dict(self, X_batch, vh_rand=False, pll_rand=False, training=False):
        feed_dict = {}
        feed_dict['input_data/X_batch:0'] = X_batch
        if vh_rand:
            feed_dict['input_data/h_rand:0'] = self._rng.rand(X_batch.shape[0], self.n_hidden)
            feed_dict['input_data/v_rand:0'] = self._rng.rand(X_batch.shape[0], self.n_visible)
        if pll_rand:
            feed_dict['input_data/pll_rand:0'] = self._rng.randint(self.n_visible, size=X_batch.shape[0])
        if training:
            self.learning_rate = next(self._learning_rate_gen)
            feed_dict['input_data/learning_rate:0'] = self.learning_rate
            self.momentum = next(self._momentum_gen)
            feed_dict['input_data/momentum:0'] = self.momentum
        return feed_dict

    def _train_epoch(self, X):
        train_msres = []
        train_plls = []
        for X_batch in (tbatch_iter if self.verbose else batch_iter)(X, self.batch_size):
            self.iter += 1
            if self.iter % self.compute_train_metrics_every_iter == 0:
                _, train_s, train_msre, pll = \
                    self._tf_session.run([self._train_op,
                                          self._tf_merged_summaries,
                                          self._msre,
                                          self._pseudo_loglik],
                                         feed_dict=self._make_tf_feed_dict(X_batch,
                                                                           vh_rand=True,
                                                                           pll_rand=True,
                                                                           training=True))
                self._tf_train_writer.add_summary(train_s, self.iter)
                train_msres.append(train_msre)
                train_plls.append(pll)
            else:
                self._tf_session.run(self._train_op,
                                     feed_dict=self._make_tf_feed_dict(X_batch,
                                                                       vh_rand=True,
                                                                       training=True))
        if not train_msres:
            return None, None
        return np.mean(train_msres), np.mean(train_plls)

    def _run_val_metrics(self, X_val):
        val_msres = []
        val_plls = []
        for X_vb in batch_iter(X_val, batch_size=self.batch_size):
            val_msre, val_pll = self._tf_session.run([self._msre, self._pseudo_loglik],
                                                     feed_dict=self._make_tf_feed_dict(X_vb,
                                                                                       vh_rand=True,
                                                                                       pll_rand=True))
            val_msres.append(val_msre)
            val_plls.append(val_pll)
        mean_msre = np.mean(val_msres)
        mean_pll = np.mean(val_plls)
        val_s = summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag='msre',
                                                                     simple_value=mean_msre),
                                           summary_pb2.Summary.Value(tag='pseudo_loglik',
                                                                     simple_value=mean_pll)])
        self._tf_val_writer.add_summary(val_s, self.iter)
        return mean_msre, mean_pll

    def _run_dfe(self, X, X_val):
        """Calculate difference between average free energies of subsets
        of validation and training sets to monitor overfitting,
        as proposed in [2]. If the model is not overfitting at all, this
        quantity should be close to zero. Once this value starts
        growing, the model is overfitting and the value represent the amount
        of overfitting.
        """
        self._free_energy_op = tf.get_collection('free_energy_op')[0]
        train_fes, val_fes = [], []
        for _, X_b in zip(xrange(self.n_batches_for_dfe),
                          batch_iter(X, batch_size=self.batch_size)):
            train_fe = self._tf_session.run(self._free_energy_op,
                                            feed_dict=self._make_tf_feed_dict(X_b))
            train_fes.append(train_fe)
        for _, X_vb in zip(xrange(self.n_batches_for_dfe),
                           batch_iter(X_val, batch_size=self.batch_size)):
            val_fe = self._tf_session.run(self._free_energy_op,
                                          feed_dict=self._make_tf_feed_dict(X_vb))
            val_fes.append(val_fe)
        dfe = np.mean(val_fes) - np.mean(train_fes)
        dfe_s = summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag='dfe',
                                                                     simple_value=dfe)])
        self._tf_val_writer.add_summary(dfe_s, self.iter)
        return dfe

    def _fit(self, X, X_val=None):
        self._learning_rate_gen = make_inf_generator(self.learning_rate)
        self._momentum_gen = make_inf_generator(self.momentum)
        self._train_op = tf.get_collection('train_op')[0]
        self._msre = tf.get_collection('msre')[0]
        self._pseudo_loglik = tf.get_collection('pseudo_loglik')[0]
        while self.epoch < self.max_epoch:
            val_msre = val_pll = None
            dfe = None
            self.epoch += 1
            train_msre, train_pll = self._train_epoch(X)
            if X_val is not None and self.epoch % self.compute_val_metrics_every_epoch == 0:
                val_msre, val_pll = self._run_val_metrics(X_val)
            if X_val is not None and self.epoch % self.compute_dfe_every_epoch == 0:
                dfe = self._run_dfe(X, X_val)
            if self.verbose:
                s = "epoch: {0:{1}}/{2}"\
                    .format(self.epoch, len(str(self.max_epoch)), self.max_epoch)
                if train_msre is not None: s += " ; msre: {0:.4f}".format(train_msre)
                if train_pll is not None: s += " ; pll: {0:.3f}".format(train_pll)
                if val_msre is not None: s += " ; val.msre: {0:.4f}".format(val_msre)
                if val_pll is not None: s += " ; val.pll: {0:.3f}".format(val_pll)
                if dfe is not None: s += " ; dfe: {0:.2f}".format(dfe)
                print s
            self._save_model(global_step=self.epoch)

    @run_in_tf_session
    def transform(self, X):
        """Compute hidden units activation probabilities."""
        self._transform_op = tf.get_collection('transform_op')[0]
        H = np.zeros((len(X), self.n_hidden))
        start = 0
        for X_b in batch_iter(X, batch_size=self.batch_size):
            H_b = self._transform_op.eval(feed_dict=self._make_tf_feed_dict(X_b, vh_rand=True))
            H[start:(start + self.batch_size)] = H_b
            start += self.batch_size
        return H
