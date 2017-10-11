import numpy as np
import tensorflow as tf
from tensorflow.core.framework import summary_pb2

from base import TensorFlowModel, run_in_tf_session
from utils import (make_list_from, batch_iter, epoch_iter,
                   write_during_training)


class DBM(TensorFlowModel):
    """Deep Boltzmann Machine.

    Parameters
    ----------
    rbms : [BaseRBM]
        Array of already pretrained RBMs going from visible units
        to the most hidden ones.
    n_particles : positive int
        Number of persistent Markov chains (i.e., "fantasy particles").
    n_gibbs_steps : positive int or iterable
        Number of Gibbs steps for PCD. Values are updated after each epoch.
    max_mf_updates : positive int
        Maximum number of mean-field updates per weight update.
    mf_tol : positive float
        Mean-field tolerance.
    learning_rate, momentum : positive float or iterable
        Gradient descent parameters. Values are updated after each epoch.
    max_epoch : positive int
        Train till this epoch.
    batch_size : positive int
        Input batch size for training. Total number of training examples should
        be divisible by this number.
    l2 : non-negative float
        L2 weight decay coefficient.
    max_norm : positive float
        Maximum norm constraint. Might be useful to use for this model instead
        of L2 weight decay as recommended in [3]

    References
    ----------
    [1] Salakhutdinov, R. and Hinton, G. (2009). Deep Boltzmann machines.
        In AISTATS 2009
    [2] Salakhutdinov, R. Learning Deep Boltzmann Machines, Matlab code.
        url: https://www.cs.toronto.edu/~rsalakhu/DBM.html
    [3] Goodfellow, I. et. al. (2013). Joint Training of Deep Boltzmann machines
        for Classification.
    """
    def __init__(self, rbms=None,
                 n_particles=100, v_particle_init=None, h_particles_init=None,
                 n_gibbs_steps=1, max_mf_updates=10, mf_tol=1e-7,
                 learning_rate=0.0005, momentum=0.9, max_epoch=10, batch_size=100,
                 L2=0., max_norm=np.inf,
                 sample_v_states=True, sample_h_states=None,
                 train_metrics_every_iter=10, val_metrics_every_epoch=1,
                 verbose=False, save_after_each_epoch=False,
                 model_path='dbm_model/', *args, **kwargs):
        super(DBM, self).__init__(model_path=model_path, *args, **kwargs)
        self.n_layers = None
        self.n_visible = None
        self.n_hiddens = None
        self.load_rbms(rbms)

        self.n_particles = n_particles
        self._v_particle_init = v_particle_init
        self._h_particles_init = h_particles_init

        self.n_gibbs_steps = make_list_from(n_gibbs_steps)
        self.max_mf_updates = max_mf_updates
        self.mf_tol = mf_tol

        self.learning_rate = make_list_from(learning_rate)
        self.momentum = make_list_from(momentum)
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.L2 = L2
        self.max_norm = max_norm

        self.sample_v_states = sample_v_states
        self.sample_h_states = sample_h_states or [True] * self.n_hiddens

        self.train_metrics_every_iter = train_metrics_every_iter
        self.val_metrics_every_epoch = val_metrics_every_epoch
        self.verbose = verbose
        self.save_after_each_epoch = save_after_each_epoch

        # current epoch and iter
        self.epoch = 0
        self.iter = 0

        # tf constants
        self._n_particles = None
        self._max_mf_updates = None
        self._mf_tol = None
        self._batch_size = None
        self._L2 = None
        self._max_norm = None
        self._N = None
        self._M = None

        # tf input data
        self._learning_rate = None
        self._momentum = None
        self._n_gibbs_steps = None
        self._X_batch = None

        # tf vars
        self._W = []
        self._vb = None
        self._hb = []

        self._dW = []
        self._dvb = None
        self._dhb = []

        self._mu = []
        self._mu_new = []
        self._v = None
        self._v_new = None
        self._H = []
        self._H_new = []

        # tf operations
        self._train_op = None
        self._transform_op = None
        self._msre = None
        self._n_mf_updates = None
        self._sample_v_particle = None

    def load_rbms(self, rbms):
        self._rbms = rbms

        # create some shortcuts
        self.n_layers = len(self._rbms)
        # TODO: remove this assertion
        assert self.n_layers >= 2
        self.n_visible = self._rbms[0].n_visible
        self.n_hiddens = [rbm.n_hidden for rbm in self._rbms]

        # extract weights and biases
        self._W_init, self._vb_init, self._hb_init = [], [], []
        for i in xrange(self.n_layers):
            weights = self._rbms[i].get_tf_params(scope='weights')
            self._W_init.append(weights['W'])
            self._vb_init.append(weights['vb'])
            self._hb_init.append(weights['hb'])

        # collect resp. layers of units
        self._v_layer = self._rbms[0]._v_layer
        self._h_layers = [rbm._h_layer for rbm in self._rbms]

        # ... and update their dtypes
        self._v_layer.tf_dtype = self._tf_dtype
        for h in self._h_layers:
            h.tf_dtype = self._tf_dtype

    def _make_constants(self):
        with tf.name_scope('constants'):
            self._n_particles = tf.constant(self.n_particles, dtype=tf.int32, name='n_particles')
            self._max_mf_updates = tf.constant(self.max_mf_updates,
                                                        dtype=tf.int32, name='max_mf_updates')
            self._mf_tol = tf.constant(self.mf_tol, dtype=self._tf_dtype, name='mf_tol')
            self._batch_size = tf.constant(self.batch_size, dtype=tf.int32, name='batch_size')
            self._L2 = tf.constant(self.L2, dtype=self._tf_dtype, name='L2_coef')
            self._max_norm = tf.constant(self.max_norm, dtype=self._tf_dtype, name='max_norm_coef')
            self._N = tf.cast(self._batch_size, dtype=self._tf_dtype, name='N')
            self._M = tf.cast(self._n_particles, dtype=self._tf_dtype, name='M')

    def _make_placeholders(self):
        with tf.name_scope('input_data'):
            self._learning_rate = tf.placeholder(self._tf_dtype, [], name='learning_rate')
            self._momentum = tf.placeholder(self._tf_dtype, [], name='momentum')
            self._n_gibbs_steps = tf.placeholder(tf.int32, [], name='n_gibbs_steps')
            self._X_batch = tf.placeholder(self._tf_dtype, [None, self.n_visible], name='X_batch')

    def _make_vars(self):
        # Compose weights and biases of DBM from trained RBMs' ones
        # and account double-counting evidence problem [1].
        # Initialize intermediate biases as mean of current RBM's
        # hidden biases and visible ones of the next, as in [2]
        W_init, hb_init = [], []
        vb_init = self._vb_init[0]
        for i in xrange(self.n_layers):
            W = self._W_init[i]
            vb = self._vb_init[i]
            hb = self._hb_init[i]

            if i in (0, self.n_layers - 1):
                W *= 0.5  # equivalent to training with RBMs with doubled weights
            W_init.append(W)
            if i < self.n_layers - 1:
                hb *= 0.5
            hb_init.append(hb)
            if i > 0:
                hb_init[i - 1] += 0.5 * vb

        # initialize weights and biases
        with tf.name_scope('weights'):
            t = tf.constant(vb_init, dtype=self._tf_dtype, name='vb_init')
            self._vb = tf.Variable(t, dtype=self._tf_dtype, name='vb')
            tf.summary.histogram('vb_hist', self._vb)

            for i in xrange(self.n_layers):
                T = tf.constant(W_init[i], dtype=self._tf_dtype, name='W_init')
                W = tf.Variable(T, dtype=self._tf_dtype, name='W')
                self._W.append(W)
                tf.summary.histogram('W_hist', W)

            for i in xrange(self.n_layers):
                t = tf.constant(hb_init[i],  dtype=self._tf_dtype, name='hb_init')
                hb = tf.Variable(t,  dtype=self._tf_dtype, name='hb')
                self._hb.append(hb)
                tf.summary.histogram('hb_hist', hb)

        # initialize grads accumulators
        with tf.name_scope('weights_updates'):
            t = tf.zeros(vb_init.shape, dtype=self._tf_dtype, name='dvb_init')
            self._dvb = tf.Variable(t, name='dvb')
            tf.summary.histogram('dvb_hist', self._dvb)

            for i in xrange(self.n_layers):
                T = tf.zeros(W_init[i].shape, dtype=self._tf_dtype, name='dW_init')
                dW = tf.Variable(T, name='dW')
                tf.summary.histogram('dW_hist', dW)
                self._dW.append(dW)

            for i in xrange(self.n_layers):
                t = tf.zeros(hb_init[i].shape, dtype=self._tf_dtype, name='dhb_init')
                dhb = tf.Variable(t, name='dhb')
                tf.summary.histogram('dhb_hist', dhb)
                self._dhb.append(dhb)

        # initialize variational parameters
        with tf.name_scope('variational_params'):
            for i in xrange(self.n_layers):
                with tf.name_scope('mu'):
                    t = tf.zeros([self._batch_size, self.n_hiddens[i]], dtype=self._tf_dtype)
                    mu = tf.Variable(t, name='mu')
                    t_new = tf.zeros([self._batch_size, self.n_hiddens[i]], dtype=self._tf_dtype)
                    mu_new = tf.Variable(t_new, name='mu_new')
                    tf.summary.histogram('mu', mu)
                    self._mu.append(mu)
                    self._mu_new.append(mu_new)

        # initialize fantasy particles
        with tf.name_scope('fantasy_particles'):
            if self._v_particle_init is not None:
                t = tf.constant(self._v_particle_init, dtype=self._tf_dtype, name='v_init')
            else:
                t = self._v_layer.init(batch_size=self._n_particles)
            self._v = tf.Variable(t, dtype=self._tf_dtype, name='v')
            t_new = self._v_layer.init(batch_size=self._n_particles)
            self._v_new = tf.Variable(t_new, dtype=self._tf_dtype, name='v_new')

            for i in xrange(self.n_layers):
                with tf.name_scope('h_particle'):
                    if self._h_particles_init is not None:
                        q = tf.constant(self._h_particles_init[i], shape=[self.n_particles, self.n_hiddens[i]],
                                        dtype=self._tf_dtype, name='h_init')
                    else:
                        q = self._h_layers[i].init(batch_size=self._n_particles)
                    h = tf.Variable(q, dtype=self._tf_dtype, name='h')
                    q_new = self._h_layers[i].init(batch_size=self._n_particles)
                    h_new = tf.Variable(q_new, dtype=self._tf_dtype, name='h_new')
                    self._H.append(h)
                    self._H_new.append(h_new)

    def _make_gibbs_step(self, v, H, v_new, H_new, update_v=True, sample=True):
        """Compute one Gibbs step."""
        with tf.name_scope('gibbs_step'):

            # update first hidden layer
            with tf.name_scope('means_h0_hat_given_v_h1'):
                T1 = tf.matmul(v, self._W[0])
                T2 = tf.matmul(a=H[1], b=self._W[1], transpose_b=True)
                H_new[0] = self._h_layers[0].activation(T1 + T2, self._hb[0])
            if sample and self.sample_h_states[0]:
                with tf.name_scope('sample_h0_hat_given_v_h1'):
                    H_new[0] = self._h_layers[0].sample(means=H_new[0])

            # update the intermediate hidden layers if any
            for i in xrange(1, self.n_layers - 1):
                with tf.name_scope('means_h{0}_hat_given_h{1}_hat_h{2}'.format(i, i - 1, i + 1)):
                    T1 = tf.matmul(H_new[i - 1], self._W[i])
                    T2 = tf.matmul(a=H[i + 1], b=self._W[i + 1], transpose_b=True)
                    H_new[i] = self._h_layers[i].activation(T1 + T2, self._hb[i])
                if sample and self.sample_h_states[i]:
                    with tf.name_scope('sample_h{0}_hat_given_h{1}_hat_h{2}'.format(i, i - 1, i + 1)):
                        H_new[i] = self._h_layers[i].sample(means=H_new[i])

            # update last hidden layer
            with tf.name_scope('means_h{0}_hat_given_h{1}_hat'.format(self.n_layers - 1, self.n_layers - 2)):
                T = tf.matmul(H_new[-2], self._W[-1])
                H_new[-1] = self._h_layers[-1].activation(T, self._hb[-1])
            if sample and self.sample_h_states[-1]:
                with tf.name_scope('sample_h{0}_hat_given_h{1}_hat'.format(self.n_layers - 1, self.n_layers - 2)):
                    H_new[-1] = self._h_layers[-1].sample(means=H_new[-1])

            # update visible layer
            if update_v:
                with tf.name_scope('means_v_hat_given_h0_hat'):
                    T = tf.matmul(a=H_new[0], b=self._W[0], transpose_b=True)
                    v_new = self._v_layer.activation(T, self._vb)
                if sample and self.sample_v_states:
                    with tf.name_scope('sample_v_hat_given_h_hat'):
                        v_new = self._v_layer.sample(means=v_new)

        return v, H, v_new, H_new

    def _make_mf(self):
        """Run mean-field updates for current mini-batch"""
        with tf.name_scope('mean_field'):
            # randomly initialize mu_new
            init_ops = []
            for i in xrange(self.n_layers):
                q = self._h_layers[i].init(self.batch_size)
                init_op = tf.assign(self._mu_new[i], q, name='init_mu_new')
                init_ops.append(init_op)

            # run mean-field updates until convergence
            def mf_cond(step, max_step, tol, X_batch, mu, mu_new):
                c1 = step < max_step
                c2 = tf.reduce_max([tf.norm(u - v, ord=np.inf) for u, v in zip(mu, mu_new)]) > tol
                return tf.logical_and(c1, c2)

            def mf_body(step, max_step, tol, X_batch, mu, mu_new):
                _, mu, _, mu_new = self._make_gibbs_step(X_batch, mu, X_batch, mu_new,
                                                         update_v=False, sample=False)
                return step + 1, max_step, tol, X_batch, mu_new, mu  # swap mu and mu_new

            with tf.control_dependencies(init_ops):  # make sure mu's are initialized
                n_mf_updates, _, _, _, mu, _ = tf.while_loop(cond=mf_cond, body=mf_body,
                                                             loop_vars=[tf.constant(0),
                                                                        self._max_mf_updates,
                                                                        self._mf_tol,
                                                                        self._X_batch,
                                                                        self._mu, self._mu_new],
                                                             back_prop=False,
                                                             name='mean_field_updates')
            return n_mf_updates, mu

    def _make_particles_update(self, n_steps=None):
        """Update fantasy particles by running Gibbs sampler
        for specified number of steps.
        """
        if n_steps is None:
            n_steps = self._n_gibbs_steps
        with tf.name_scope('gibbs_chain'):
            def sa_cond(step, max_step, v, H, v_new, H_new):
                return step < max_step

            def sa_body(step, max_step, v, H, v_new, H_new):
                v, H, v_new, H_new = self._make_gibbs_step(v, H, v_new, H_new,
                                                           update_v=True, sample=True)
                return step + 1, max_step, v_new, H_new, v, H # swap particles

            _, _, v, H, v_new, H_new = tf.while_loop(cond=sa_cond, body=sa_body,
                                                     loop_vars=[tf.constant(0),
                                                                n_steps,
                                                                self._v, self._H,
                                                                self._v_new, self._H_new],
                                                     back_prop=False)
            v_update = self._v.assign(v)
            v_new_update = self._v_new.assign(v_new)
            H_updates = [ self._H[i].assign(H[i]) for i in xrange(self.n_layers) ]
            H_new_updates = [ self._H_new[i].assign(H_new[i]) for i in xrange(self.n_layers) ]
        return v_update, H_updates, v_new_update, H_new_updates

    def _apply_max_norm(self, T):
        n = tf.norm(T, axis=0)
        return T * tf.minimum(self._max_norm, n) / tf.maximum(n, 1e-8)

    def _make_train_op(self):
        # run mean-field updates for current mini-batch
        n_mf_updates, mu = self._make_mf()

        # encoded data, used by the transform method
        with tf.name_scope('transform'):
            transform_op = tf.identity(mu[-1])
            tf.add_to_collection('transform_op', transform_op)

        # update fantasy particles by running Gibbs sampler
        # for specified number of steps
        v_update, H_updates, v_new_update, H_new_updates = self._make_particles_update()

        with tf.control_dependencies([v_update, v_new_update] + H_updates + H_new_updates +\
                                     [self._mu[i].assign(mu[i]) for i in xrange(self.n_layers)]):
            # compute gradients estimates (= positive - negative associations)
            with tf.name_scope('grads_estimates'):
                # visible bias
                with tf.name_scope('dvb'):
                    dvb = tf.reduce_mean(self._X_batch, axis=0) - tf.reduce_mean(self._v, axis=0)

                dW = []
                # first layer of weights
                with tf.name_scope('dW'):
                    dW_0_positive = tf.matmul(a=self._X_batch, b=mu[0], transpose_a=True) / self._N
                    dW_0_negative = tf.matmul(a=self._v, b=self._H[0], transpose_a=True) / self._M
                    dW_0 = (dW_0_positive - dW_0_negative) - self._L2 * self._W[0]
                    dW.append(dW_0)
                # ... rest of them
                for i in xrange(1, self.n_layers):
                    with tf.name_scope('dW'):
                        dW_i_positive = tf.matmul(a=mu[i - 1], b=mu[i], transpose_a=True) / self._N
                        dW_i_negative = tf.matmul(a=self._H[i - 1], b=self._H[i], transpose_a=True) / self._M
                        dW_i = (dW_i_positive - dW_i_negative) - self._L2 * self._W[i]
                        dW.append(dW_i)

                dhb = []
                # hidden biases
                for i in xrange(self.n_layers):
                    with tf.name_scope('dhb'):
                        dhb_i = tf.reduce_mean(mu[i], axis=0) - tf.reduce_mean(self._H[i], axis=0)
                        dhb.append(dhb_i)

            # update parameters
            with tf.name_scope('momentum_updates'):
                with tf.name_scope('dvb'):
                    dvb_update = self._dvb.assign(self._learning_rate * (self._momentum * self._dvb + dvb))
                    vb_update = self._vb.assign_add(dvb_update)

                W_updates = []
                for i in xrange(self.n_layers):
                    with tf.name_scope('dW'):
                        dW_update = self._dW[i].assign(self._learning_rate * (self._momentum * self._dW[i] + dW[i]))
                        W_update = self._W[i].assign_add(dW_update)
                        W_update = self._W[i].assign(self._apply_max_norm(W_update))
                        W_updates.append(W_update)

                hb_updates = []
                for i in xrange(self.n_layers):
                    with tf.name_scope('dhb'):
                        dhb_update = self._dhb[i].assign(self._learning_rate * (self._momentum * self._dhb[i] + dhb[i]))
                        hb_update = self._hb[i].assign_add(dhb_update)
                        hb_updates.append(hb_update)

            # assemble train_op
            with tf.name_scope('training_step'):
                particles_updates = tf.group(v_update, v_new_update,
                                             tf.group(*H_updates), tf.group(*H_new_updates),
                                             name='particles_updates')
                weights_updates = tf.group(vb_update,
                                           tf.group(*W_updates),
                                           tf.group(*hb_updates),
                                           name='weights_update')
                train_op = tf.group(weights_updates, particles_updates)
                tf.add_to_collection('train_op', train_op)

        # compute metrics
        with tf.name_scope('mean_squared_reconstruction_error'):
            T = tf.matmul(a=mu[0], b=self._W[0], transpose_b=True)
            v_means = self._v_layer.activation(T, self._vb)
            v_means = tf.identity(v_means, name='x_reconstruction')
            msre = tf.reduce_mean(tf.square(self._X_batch - v_means))
            tf.add_to_collection('msre', msre)

        tf.add_to_collection('n_mf_updates', n_mf_updates)

        # collect summaries
        tf.summary.scalar('mean_squared_recon_error', msre)
        tf.summary.scalar('n_mf_updates', n_mf_updates)

    def _make_sample_v_particle(self):
        with tf.name_scope('sample_v_particle'):
            v_update, H_updates, v_new_update, H_new_updates = \
                self._make_particles_update(n_steps=self._n_gibbs_steps)
            with tf.control_dependencies([v_update, v_new_update] + H_updates + H_new_updates):
                T = tf.matmul(a=self._H[0], b=self._W[0], transpose_b=True)
                v_probs = self._v_layer.activation(T, self._vb)
                sample_v_particle = self._v.assign(v_probs)
        tf.add_to_collection('sample_v_particle', sample_v_particle)

    def _make_tf_model(self):
        self._make_constants()
        self._make_placeholders()
        self._make_vars()
        self._make_train_op()
        self._make_sample_v_particle()

    def _make_tf_feed_dict(self, X_batch=None, n_gibbs_steps=None):
        d = {}
        d['learning_rate'] = self.learning_rate[min(self.epoch, len(self.learning_rate) - 1)]
        d['momentum'] = self.momentum[min(self.epoch, len(self.momentum) - 1)]
        if X_batch is not None:
            d['X_batch'] = X_batch
        d['n_gibbs_steps'] = self.n_gibbs_steps[min(self.epoch, len(self.n_gibbs_steps) - 1)] \
                             if n_gibbs_steps is None else n_gibbs_steps
        # prepend name of the scope, and append ':0'
        feed_dict = {}
        for k, v in d.items():
            feed_dict['input_data/{0}:0'.format(k)] = v
        return feed_dict

    def _train_epoch(self, X):
        train_msres, train_n_mf_updates = [], []
        for X_batch in batch_iter(X, self.batch_size, verbose=self.verbose):
            self.iter += 1
            if self.iter % self.train_metrics_every_iter == 0:
                msre, n_mf_upds, _, s = self._tf_session.run([self._msre, self._n_mf_updates,
                                                              self._train_op, self._tf_merged_summaries],
                                                             feed_dict=self._make_tf_feed_dict(X_batch))
                train_msres.append(msre)
                train_n_mf_updates.append(n_mf_upds)
                self._tf_train_writer.add_summary(s, self.iter)
            else:
                self._tf_session.run(self._train_op,
                                     feed_dict=self._make_tf_feed_dict(X_batch))
        return (np.mean(train_msres) if train_msres else None,
                np.mean(train_n_mf_updates) if train_n_mf_updates else None)

    def _run_val_metrics(self, X_val):
        val_msres, val_n_mf_updates = [], []
        for X_vb in batch_iter(X_val, batch_size=self.batch_size):
            msre, n_mf_upds = self._tf_session.run([self._msre, self._n_mf_updates],
                                                   feed_dict=self._make_tf_feed_dict(X_vb))
            val_msres.append(msre)
            val_n_mf_updates.append(n_mf_upds)
        mean_msre = np.mean(val_msres)
        mean_n_mf_updates = np.mean(val_n_mf_updates)
        s = summary_pb2.Summary(value=[
            summary_pb2.Summary.Value(tag='mean_squared_recon_error', simple_value=mean_msre),
            summary_pb2.Summary.Value(tag='n_mf_updates', simple_value=mean_n_mf_updates),
        ])
        self._tf_val_writer.add_summary(s, self.iter)
        return mean_msre, mean_n_mf_updates

    def _fit(self, X, X_val=None):
        # load ops requested
        self._train_op = tf.get_collection('train_op')[0]
        self._msre = tf.get_collection('msre')[0]
        self._n_mf_updates = tf.get_collection('n_mf_updates')[0]

        # main loop
        val_msre, val_n_mf_updates = None, None
        for self.epoch in epoch_iter(start_epoch=self.epoch, max_epoch=self.max_epoch,
                                     verbose=self.verbose):
            train_msre, train_n_mf_updates = self._train_epoch(X)

            # run validation metrics if needed
            if X_val is not None and self.epoch % self.val_metrics_every_epoch == 0:
                val_msre, val_n_mf_updates = self._run_val_metrics(X_val)

            # print progress
            if self.verbose:
                s = "epoch: {0:{1}}/{2}".format(self.epoch, len(str(self.max_epoch)), self.max_epoch)
                if train_msre:
                    s += "; msre: {0:.5f}".format(train_msre)
                if train_n_mf_updates:
                    s += "; n_mf_upds: {0:.2f}".format(train_n_mf_updates)
                if val_msre:
                    s += "; val.msre: {0:.5f}".format(val_msre)
                if val_n_mf_updates:
                    s += "; val.n_mf_upds: {0:.2f}".format(val_n_mf_updates)
                write_during_training(s)

            # save if needed
            if self.save_after_each_epoch:
                self._save_model(global_step=self.epoch)

    @run_in_tf_session()
    def transform(self, X):
        """Compute hidden units' (from last layer) activation probabilities."""
        self._transform_op = tf.get_collection('transform_op')[0]
        Q = np.zeros((len(X), self.n_hiddens[-1]))
        start = 0
        for X_b in batch_iter(X, batch_size=self.batch_size, verbose=self.verbose):
            Q_b = self._transform_op.eval(feed_dict=self._make_tf_feed_dict(X_b))
            Q[start:(start + self.batch_size)] = Q_b
            start += self.batch_size
        return Q

    @run_in_tf_session(update_seed=True)
    def sample_v_particle(self, n_gibbs_steps=0, save_model=False):
        if not self.called_fit:
            raise RuntimeError('`fit` must be called before calling `sample_v_particle`')
        self._sample_v_particle = tf.get_collection('sample_v_particle')[0]
        v = self._tf_session.run(self._sample_v_particle,
                                 feed_dict=self._make_tf_feed_dict(n_gibbs_steps=n_gibbs_steps))
        if save_model:
            self._save_model()
        return v

    def _serialize(self, params):
        for k, v in params.items():
            if isinstance(v, np.ndarray):
                # noinspection PyUnresolvedReferences
                params[k] = v.tolist()
        return params
