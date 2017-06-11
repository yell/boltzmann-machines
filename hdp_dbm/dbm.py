import numpy as np
import tensorflow as tf
from tensorflow.core.framework import summary_pb2

from base import TensorFlowModel, run_in_tf_session
from utils import (batch_iter, epoch_iter,
                   make_inf_generator, print_inline)


class DBM(TensorFlowModel):
    """Deep Boltzmann Machine.

    Parameters
    ----------
    rbms : [BaseRBM]
        Array of already pretrained RBMs going from visible units
        to the most hidden ones.
    n_particles : int
        Number of persistent Markov chains (i.e., "fantasy particles").
    max_mf_updates_per_iter : int
        Maximum number of mean-field to perform on each iteration.
    mf_tol : float


    References
    ----------
    [1] Salakhutdinov, R. and Hinton, G. (2009). Deep Boltzmann machines.
        In AISTATS 2009
    [2] Salakhutdinov, R. Learning Deep Boltzmann Machines, Matlab code.
        url: https://www.cs.toronto.edu/~rsalakhu/DBM.html
    """
    def __init__(self, rbms=None, v_particle_init=None, h_particles_init=None,
                 n_particles=100, n_particles_updates_per_iter=5,
                 max_mf_updates_per_iter=10, mf_tol=1e-7,
                 learning_rate=0.001, momentum=0.9, max_epoch=10, batch_size=100, L2=1e-5,
                 train_metrics_every_iter=10, val_metrics_every_epoch=1,
                 verbose=False, save_after_each_epoch=False,
                 model_path='dbm_model/', *args, **kwargs):
        super(DBM, self).__init__(model_path=model_path, *args, **kwargs)
        self.load_rbms(rbms)
        self._v_particle_init = v_particle_init
        self._h_particles_init = h_particles_init

        self.n_particles = n_particles
        self.n_particles_updates_per_iter = n_particles_updates_per_iter
        self.max_mf_updates_per_iter = max_mf_updates_per_iter
        self.mf_tol = mf_tol

        self.learning_rate = learning_rate
        self._learning_rate_gen = None
        self.momentum = momentum
        self._momentum_gen = None
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.L2 = L2

        self.train_metrics_every_iter = train_metrics_every_iter
        self.val_metrics_every_epoch = val_metrics_every_epoch
        self.verbose = verbose
        self.save_after_each_epoch = save_after_each_epoch

        # current epoch and iter
        self.epoch = 0
        self.iter = 0

        # tf constants
        self._L2 = None
        self._n_particles_updates_per_iter = None
        self._n_particles = None
        self._batch_size = None
        self._max_mf_updates_per_iter = None
        self._mf_tol = None
        self._N = None
        self._M = None

        # tf input data
        self._X_batch = None
        self._learning_rate = None
        self._momentum = None
        self._v_rand = None
        self._h_rand = []
        self._n_gibbs_steps = None

        # tf vars
        self._W = []
        self._hb = []
        self._vb = None

        self._dW = []
        self._dhb = []
        self._dvb = None

        self._mu = []
        self._mu_new = []
        self._v = None
        self._H = []
        self._v_new = None
        self._H_new = []

        # tf operations
        self._train_op = None
        self._msre = None
        self._n_mf_updates = None
        self._sample_v_particle = None

    def load_rbms(self, rbms):
        self._rbms = rbms
        if self._rbms is not None:

            # create some shortcuts
            self.n_layers = len(self._rbms)
            assert self.n_layers >= 2
            self.n_visible = self._rbms[0].n_visible
            self.n_hiddens = [rbm.n_hidden for rbm in self._rbms]

            # extract weights and biases
            self._W_tmp, self._vb_tmp, self._hb_tmp = [], [], []
            for i in xrange(self.n_layers):
                weights = self._rbms[i].get_tf_params(scope='weights')
                self._W_tmp.append(weights['W'])
                self._vb_tmp.append(weights['vb'])
                self._hb_tmp.append(weights['hb'])

            # collect resp. layers of units
            self._v_layer = self._rbms[0]._v_layer
            self._h_layers = [rbm._h_layer for rbm in self._rbms]

            # ... and update their dtypes
            self._v_layer.tf_dtype = self._tf_dtype
            for h in self._h_layers: h.tf_dtype = self._tf_dtype
        else:
            self.n_layers = None
            self.n_visible = None
            self.n_hiddens = None

    def _make_constants(self):
        with tf.name_scope('constants'):
            self._L2 = tf.constant(self.L2, dtype=self._tf_dtype, name='L2_coef')
            self._n_particles = tf.constant(self.n_particles, dtype=tf.int32, name='n_particles')
            self._n_particles_updates_per_iter = \
                tf.constant(self.n_particles_updates_per_iter, dtype=tf.int32, name='n_particles_updates_per_iter')
            self._batch_size = tf.constant(self.batch_size, dtype=tf.int32, name='batch_size')
            self._max_mf_updates_per_iter = tf.constant(self.max_mf_updates_per_iter,
                                                        dtype=tf.int32, name='max_mf_updates_per_iter')
            self._mf_tol = tf.constant(self.mf_tol, dtype=self._tf_dtype, name='mf_tol')
            self._N = tf.cast(self._batch_size, dtype=self._tf_dtype)
            self._M = tf.cast(self._n_particles, dtype=self._tf_dtype)

    def _make_placeholders(self):
        with tf.name_scope('input_data'):
            self._X_batch = tf.placeholder(self._tf_dtype, [None, self.n_visible], name='X_batch')
            self._learning_rate = tf.placeholder(self._tf_dtype, [], name='learning_rate')
            self._momentum = tf.placeholder(self._tf_dtype, [], name='momentum')
            self._v_rand = tf.placeholder(self._tf_dtype, [None, self.n_visible], name='v_rand')
            self._h_rand = []
            for i in xrange(self.n_layers):
                P = tf.placeholder(self._tf_dtype, [None, self.n_hiddens[i]], name='h_rand')
                self._h_rand.append(P)
            self._n_gibbs_steps = tf.placeholder(tf.int32, [], name='n_gibbs_steps')

    def _make_vars(self):
        # Compose weights and biases of DBM from trained RBMs' ones
        # and account double-counting evidence problem [1].
        # Initialize intermediate biases as mean of current RBM's
        # hidden biases and visible ones of the next, as in [2]
        W_init, hb_init = [], []
        vb_init = self._vb_tmp[0]
        for i in xrange(self.n_layers):
            W = self._W_tmp[i]
            hb = self._hb_tmp[i]
            vb = self._vb_tmp[i]
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
            t = tf.constant(vb_init, name='vb_init', dtype=self._tf_dtype)
            self._vb = tf.Variable(t, name='vb', dtype=self._tf_dtype)
            tf.summary.histogram('vb_hist', self._vb)

            for i in xrange(self.n_layers):
                T = tf.constant(W_init[i], name='W_init', dtype=self._tf_dtype)
                W = tf.Variable(T, name='W', dtype=self._tf_dtype)
                self._W.append(W)
                tf.summary.histogram('W_hist', W)

            for i in xrange(self.n_layers):
                t = tf.constant(hb_init[i], name='hb_init', dtype=self._tf_dtype)
                hb = tf.Variable(t, name='hb', dtype=self._tf_dtype)
                self._hb.append(hb)
                tf.summary.histogram('hb_hist', hb)

        # initialize grads accumulators
        with tf.name_scope('grads'):
            t = tf.zeros_like(self._vb, dtype=self._tf_dtype, name='dvb_init')
            self._dvb = tf.Variable(t, name='dvb')
            tf.summary.histogram('dvb_hist', self._dvb)

            for i in xrange(self.n_layers):
                T = tf.zeros_like(self._W[i], dtype=self._tf_dtype, name='dW_init')
                dW = tf.Variable(T, name='dW')
                tf.summary.histogram('dW_hist', dW)
                self._dW.append(dW)

            for i in xrange(self.n_layers):
                t = tf.zeros_like(self._hb[i], dtype=self._tf_dtype, name='dhb_init')
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
            with tf.name_scope('v_particle'):
                if self._v_particle_init is not None:
                    t = tf.constant(self._v_particle_init, dtype=self._tf_dtype, name='v_init')
                else:
                    t = self._v_layer.init(batch_size=self._n_particles,
                                           random_seed=self.make_random_seed())
                self._v = tf.Variable(t, dtype=self._tf_dtype, name='v')
                t_new = self._v_layer.init(batch_size=self._n_particles,
                                           random_seed=self.make_random_seed())
                self._v_new = tf.Variable(t_new, dtype=self._tf_dtype, name='v_new')

            with tf.name_scope('h_particles'):
                for i in xrange(self.n_layers):
                    with tf.name_scope('h_particle'):
                        if self._h_particles_init is not None:
                            q = tf.constant(self._h_particles_init[i], shape=[self.n_particles, self.n_hiddens[i]],
                                            dtype=self._tf_dtype, name='h_init')
                        else:
                            q = self._h_layers[i].init(batch_size=self._n_particles,
                                                       random_seed=self.make_random_seed())
                        h = tf.Variable(q, dtype=self._tf_dtype, name='h')
                        q_new = self._h_layers[i].init(batch_size=self._n_particles,
                                                       random_seed=self.make_random_seed())
                        h_new = tf.Variable(q_new, dtype=self._tf_dtype, name='h_new')
                        self._H.append(h)
                        self._H_new.append(h_new)

    def _make_gibbs_step(self, v, H, v_new, H_new, update_v=True, sample=True):
        """Compute one Gibbs step."""
        with tf.name_scope('gibbs_step'):

            # update visible layer
            if update_v:
                with tf.name_scope('means_v_given_h0'):
                    T = tf.matmul(a=H[0], b=self._W[0], transpose_b=True)
                    v_new = self._v_layer.activation(T, self._vb)
                if sample:
                    with tf.name_scope('sample_v_given_h'):
                        v_new = self._v_layer.sample(rand_data=self._v_rand, means=v_new)

            # update last hidden layer
            with tf.name_scope('means_h{0}_given_h{1}'.format(self.n_layers - 1, self.n_layers - 2)):
                T = tf.matmul(H[-2], self._W[-1])
                H_new[-1] = self._h_layers[-1].activation(T, self._hb[-1])
            if sample:
                with tf.name_scope('sample_h{0}_given_h{1}'.format(self.n_layers - 1, self.n_layers - 2)):
                    H_new[-1] = self._h_layers[-1].sample(rand_data=self._h_rand[-1], means=H_new[-1])

            # update first hidden layer
            with tf.name_scope('means_h0_given_v_h1'):
                T1 = tf.matmul(v, self._W[0])
                T2 = tf.matmul(a=H[1], b=self._W[1], transpose_b=True)
                H_new[0] = self._h_layers[0].activation(T1 + T2, self._hb[0])
            if sample:
                with tf.name_scope('sample_h0_given_v_h1'):
                    H_new[0] = self._h_layers[0].sample(rand_data=self._h_rand[0], means=H_new[0])

            # update the intermediate hidden layers if any
            for i in xrange(1, self.n_layers - 1):
                with tf.name_scope('means_h{0}_given_h{1}_h{2}'.format(i, i - 1, i + 1)):
                    T1 = tf.matmul(H[i - 1], self._W[i])
                    T2 = tf.matmul(a=H[i + 1], b=self._W[i + 1], transpose_b=True)
                    H_new[i] = self._h_layers[i].activation(T1 + T2, self._hb[i])
                if sample:
                    with tf.name_scope('sample_h{0}_given_h{1}_h{2}'.format(i, i - 1, i + 1)):
                        H_new[i] = self._h_layers[i].sample(rand_data=self._h_rand[i], means=H_new[i])
        return v, H, v_new, H_new

    def _make_mf(self):
        """Run mean-files updates for current mini-batch"""
        with tf.name_scope('mean_field'):
            # randomly initialize variational parameters
            init_ops = []
            for i in xrange(self.n_layers):
                # (1) -- random initialization
                # ----------------------------
                t = self._h_layers[i].init(self.batch_size, random_seed=self.make_random_seed())
                q = self._h_layers[i].init(self.batch_size, random_seed=self.make_random_seed())
                init_op = tf.assign(self._mu[i], t, name='init_mu')
                init_op2 = tf.assign(self._mu_new[i], q, name='init_mu')
                init_ops.append(init_op)
                init_ops.append(init_op2)

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
                                                                        self._max_mf_updates_per_iter,
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
        if n_steps is None: n_steps = self._n_particles_updates_per_iter
        with tf.name_scope('gibbs_chain'):
            def sa_cond(step, max_step, v, H, v_new, H_new):
                return step < max_step

            def sa_body(step, max_step, v, H, v_new, H_new):
                v, H, v_new, H_new = self._make_gibbs_step(v, H, v_new, H_new,
                                                           update_v=True, sample=True)
                return step + 1, max_step, v_new, H_new, v, H  # swap particles

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

    def _make_train_op(self):
        # run mean-field updates for current mini-batch
        n_mf_updates, mu = self._make_mf()

        # update fantasy particles by running Gibbs sampler
        # for specified number of steps
        v_update, H_updates, v_new_update, H_new_updates = self._make_particles_update()

        with tf.control_dependencies([v_update, v_new_update] + H_updates + H_new_updates):
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

    def _make_tf_feed_dict(self, X_batch=None, training=False, n_gibbs_steps=None):
        d = {}
        d['v_rand'] = self._v_layer.make_rand(self.n_particles, self._rng)
        d['h_rand'] = self._h_layers[0].make_rand(self.n_particles, self._rng)
        for i in xrange(1, self.n_layers):
            d['h_rand_{0}'.format(i)] = self._h_layers[i].make_rand(self.n_particles, self._rng)
        if X_batch is not None:
            d['X_batch'] = X_batch
        if training:
            d['learning_rate'] = self.learning_rate
            d['momentum'] = self.momentum
        if n_gibbs_steps is not None:
            d['n_gibbs_steps'] = n_gibbs_steps
        # prepend name of the scope, and append ':0'
        feed_dict = {}
        for k, v in d.items():
            feed_dict['input_data/{0}:0'.format(k)] = v
        return feed_dict

    def _train_epoch(self, X):
        # updates hyper-parameters if needed
        self.learning_rate = next(self._learning_rate_gen)
        self.momentum = next(self._momentum_gen)

        train_msres, train_n_mf_updates = [], []
        for X_batch in batch_iter(X, self.batch_size, verbose=self.verbose):
            self.iter += 1
            if self.iter % self.train_metrics_every_iter == 0:
                msre, n_mf_upds, _, s = self._tf_session.run([self._msre, self._n_mf_updates,
                                                              self._train_op, self._tf_merged_summaries],
                                                             feed_dict=self._make_tf_feed_dict(X_batch,
                                                                                               training=True))
                train_msres.append(msre)
                train_n_mf_updates.append(n_mf_upds)
                self._tf_train_writer.add_summary(s, self.iter)
            else:
                self._tf_session.run(self._train_op,
                                     feed_dict=self._make_tf_feed_dict(X_batch, training=True))
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
        # init generators
        self._learning_rate_gen = make_inf_generator(self.learning_rate)
        self._momentum_gen = make_inf_generator(self.momentum)

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
                print_inline(s + '\n')

            # save if needed
            if self.save_after_each_epoch:
                self._save_model(global_step=self.epoch)

    @run_in_tf_session
    def sample_v_particle(self, n_gibbs_steps=0, save_model=False):
        if not self.called_fit:
            raise RuntimeError('`fit` must be called before calling `sample_v_particle`')
        self._sample_v_particle = tf.get_collection('sample_v_particle')[0]
        v = self._tf_session.run(self._sample_v_particle,
                                 feed_dict=self._make_tf_feed_dict(n_gibbs_steps=n_gibbs_steps))
        self._save_model()
        return v


if __name__ == '__main__':
    from rbm import BernoulliRBM
    from hdp_dbm.utils.dataset import load_mnist
    from utils.plot_utils import plot_matrices
    import matplotlib.pyplot as plt
    X, _ = load_mnist('train', '../data/')
    X /= 255.
    X_val = X[-1000:]
    X = X[:2000]

    print X[:10].shape

    rbm1 = BernoulliRBM.load_model('../models/2_dbm_mnist_rbm_1/')
    rbm2 = BernoulliRBM.load_model('../models/2_dbm_mnist_rbm_2/')
    #
    print rbm1.get_tf_params(scope='weights')['W'][0][0]
    print rbm2.get_tf_params(scope='weights')['W'][0][0]
    #
    H = rbm1.transform(X)
    print H[:10].shape
    Z = rbm2.transform(H)
    print Z[:10].shape

    dbm = DBM(rbms=[rbm1, rbm2],
              n_particles=10,
              v_particle_init=X[:10].copy(),
              h_particles_init=(H[:10].copy(), Z[:10].copy()),
              n_particles_updates_per_iter=5, # or 5
              max_mf_updates_per_iter=100, # or 30
              mf_tol=1e-5,
              learning_rate=0.001,
              momentum=[.5] * 5 + [.9],
              max_epoch=3, # 300 or 500 for paper results
              batch_size=100,
              L2=2e-4,
              random_seed=1337,
              verbose=True,
              tf_dtype='float32',
              save_after_each_epoch=True,
              model_path='dbm/')

    # dbm = DBM.load_model('dbm/')

    # dbm.load_rbms([rbm1, rbm2])
    # print dbm._v_particle_init.shape
    # print dbm._h_particles_init[0]

    dbm.fit(X, X_val)
    print dbm.get_tf_params('weights')['W'][0][0] * 2




    # v = dbm.sample_v_particle(save_model=True)
    #
    # # print v[0][:100]
    # # v = dbm.get_tf_params('fantasy_particles/v_particle/')['v']
    # # print v[0][:100]
    #
    # # v_new = dbm.get_tf_params('fantasy_particles/v_particle/')['v_new']
    # # h = dbm.get_tf_params('fantasy_particles/h_particle/')['h']
    # # print "v"
    # # print v[0][:200]
    # # print "v_new"
    # # print v[0][:200]
    #
    # # print v[1][:15]
    # # print v[2][:15]
    # # print "H"
    # # print h[0][:10]
    # # print h[1][:10]
    # # print h[2][:10]
    # #
    # plot_matrices(v, 10, 1, shape=(28, 28))
    # plt.show()
