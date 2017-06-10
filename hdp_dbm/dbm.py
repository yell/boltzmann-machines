import numpy as np
import tensorflow as tf

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
    def __init__(self, rbms=None,
                 n_particles=100, n_gibbs_steps=5, max_mf_updates_per_iter=10, mf_tol=1e-7,
                 learning_rate=0.001, momentum=0.9, max_epoch=10, batch_size=100, L2=1e-5,
                 train_metrics_every_iter=10, val_metrics_every_epoch=1,
                 verbose=False, save_after_each_epoch=False,
                 model_path='dbm_model/', *args, **kwargs):
        super(DBM, self).__init__(model_path=model_path, *args, **kwargs)
        self._rbms = rbms

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

        # other params
        self.n_particles = n_particles
        self.n_gibbs_steps = n_gibbs_steps
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
        self._n_gibbs_steps = None
        self._n_particles = None
        self._batch_size = None

        # tf input data
        self._X_batch = None
        self._learning_rate = None
        self._momentum = None

        # tf vars
        self._W = []
        self._hb = []
        self._vb = None

        self._dW = []
        self._dhb = []
        self._dvb = None

        self._mu = []
        self._mu_new = []
        self._particles = []
        self._particles_new = []

        # tf operations
        self._train_op = None
        self._msre = None
        self._n_mf_updates = None

    def _make_constants(self):
        with tf.name_scope('constants'):
            self._L2 = tf.constant(self.L2, dtype=self._tf_dtype, name='L2_coef')
            self._n_particles = tf.constant(self.n_particles, dtype=tf.int32, name='n_particles')
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
                W *= 0.5 # equivalent to training with RBMs with doubled weights
            W_init.append(W)
            if i < self.n_layers - 1:
                hb *= 0.5
            hb_init.append(hb)
            if i > 0:
                hb_init[i - 1] += 0.5 * vb

        # initialize weights and biases
        with tf.name_scope('weights'):
            t = tf.identity(vb_init, name='vb_init')
            self._vb = tf.Variable(t, name='vb', dtype=self._tf_dtype)
            tf.summary.histogram('vb_hist', self._vb)

            for i in xrange(self.n_layers):
                T = tf.identity(W_init[i], name='W_init')
                W = tf.Variable(T, name='W', dtype=self._tf_dtype)
                tf.summary.histogram('W_hist', W)
                self._W.append(W)

            for i in xrange(self.n_layers):
                t = tf.identity(hb_init[i], name='hb_init')
                hb = tf.Variable(t, name='hb', dtype=self._tf_dtype)
                tf.summary.histogram('hb_hist', hb)
                self._hb.append(hb)

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
                t = self._v_layer.init(batch_size=self._n_particles,
                                       random_seed=self.make_random_seed())
                v = tf.Variable(t, dtype=self._tf_dtype, name='v')
                t_new = self._v_layer.init(batch_size=self._n_particles,
                                           random_seed=self.make_random_seed())
                v_new = tf.Variable(t_new, dtype=self._tf_dtype, name='v_new')
                H, H_new = [], []

            for j in xrange(self.n_layers):
                with tf.name_scope('h_particle'):
                    q = self._h_layers[j].init(batch_size=self._n_particles,
                                               random_seed=self.make_random_seed())
                    h = tf.Variable(q, dtype=self._tf_dtype, name='h')
                    q_new = self._h_layers[j].init(batch_size=self._n_particles,
                                                   random_seed=self.make_random_seed())
                    h_new = tf.Variable(q_new, dtype=self._tf_dtype, name='h_new')
                    H.append(h)
                    H_new.append(h_new)

            self._particles = (v, H)
            self._particles_new = (v_new, H_new)

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

    def _make_gibbs_chain(self, v, H, v_new, H_new, update_v=True, sample=True, n_steps=None):
        """Run Gibbs chain for specified number of steps"""
        n_steps = n_steps or self.n_gibbs_steps
        with tf.name_scope('gibbs_chain'):
            for _ in xrange(n_steps):
                v, H, v_new, H_new = self._make_gibbs_step(v, H, v_new, H_new, update_v=update_v,
                                                           sample=sample)
                v_new, H_new, v, H = v, H, v_new, H_new
        return v, H, v_new, H_new # v, H contain the most recent values

    def _make_mf(self):
        """Run mean-field updates until convergence for 1 batch."""
        with tf.name_scope('mean_field'):
            # randomly initialize variational parameters
            init_ops = []
            for i in xrange(self.n_layers):
                t = self._h_layers[i].init(self.batch_size, random_seed=self.make_random_seed())
                init_op = tf.assign(self._mu[i], t, name='init_mu')
                init_ops.append(init_op)

            # run mean-field updates until convergence
            mf_counter = tf.constant(0, dtype=tf.int32, name='mf_counter')
            def cond(step, X_batch, mu, mu_new):
                c1 = step < self._max_mf_updates_per_iter
                c2 = tf.reduce_max([ tf.norm(u - v, ord=np.inf) for u, v in zip(mu, mu_new) ]) > self._mf_tol
                return tf.logical_and(c1, c2)
            def body(step, X_batch, mu, mu_new):
                _, mu, _, mu_new = self._make_gibbs_step(X_batch, mu, X_batch, mu_new,
                                                         update_v=False, sample=False)
                return step + 1, X_batch, mu_new, mu # swap mu and mu_new

            with tf.control_dependencies(init_ops):
                n_mf_updates, _, mu, _ = tf.while_loop(cond=cond, body=body,
                                                       loop_vars=[mf_counter, self._X_batch,
                                                                  self._mu, self._mu_new],
                                                       back_prop=False,
                                                       name='mean_field_updates')
        return n_mf_updates, mu

    def _make_stochastic_approx(self):
        """Update fantasy particles by running Gibbs sampler
        for specified number of steps.
        """
        v, H = self._particles
        v_new, H_new = self._particles_new
        v, H, v_new, H_new = self._make_gibbs_chain(v, H, v_new, H_new,
                                                    update_v=True, sample=True)
        self._particles = (v, H)
        self._particles_new = (v_new, H_new)
        return v, H

    def _make_train_op(self):
        # run mean-field updates
        n_mf_updates, mu = self._make_mf()

        # update particles
        v, H = self._make_stochastic_approx()

        # compute gradients estimates (= positive - negative associations)
        with tf.name_scope('grads_estimates'):
            # visible bias
            with tf.name_scope('dvb'):
                dvb = tf.reduce_mean(self._X_batch, axis=0) - tf.reduce_mean(v, axis=0)

            dW = []
            # first layer of weights
            with tf.name_scope('dW'):
                dW_0_positive = tf.matmul(a=self._X_batch, b=mu[0], transpose_a=True) / self._N
                dW_0_negative = tf.matmul(a=v, b=H[0], transpose_a=True) / self._M
                dW_0 = (dW_0_positive - dW_0_negative) - self._L2 * self._W[0]
                dW.append(dW_0)
            # ... rest of them
            for i in xrange(1, self.n_layers):
                with tf.name_scope('dW'):
                    dW_i_positive = tf.matmul(a=mu[i - 1], b=mu[i], transpose_a=True) / self._N
                    dW_i_negative = tf.matmul(a=H[i - 1], b=H[i], transpose_a=True) / self._M
                    dW_i = (dW_i_positive - dW_i_negative) - self._L2 * self._W[i]
                    dW.append(dW_i)

            dhb = []
            # hidden biases
            for i in xrange(self.n_layers):
                with tf.name_scope('dhb'):
                    dhb_i = tf.reduce_mean(mu[i], axis=0) - tf.reduce_mean(H[i], axis=0)
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
            train_op = tf.group(vb_update, tf.group(*W_updates), tf.group(*hb_updates))
            tf.add_to_collection('train_op', train_op)

        # compute metrics
        with tf.name_scope('mean_squared_reconstruction_error'):
            T = tf.matmul(a=mu[0], b=self._W[0], transpose_b=True)
            v_means = self._v_layer.activation(T, self._vb)
            msre = tf.reduce_mean(tf.square(self._X_batch - v_means))
            tf.add_to_collection('msre', msre)

        tf.add_to_collection('n_mf_updates', n_mf_updates)

        # collect summaries
        tf.summary.scalar('mean_squared_recon_error', msre)
        tf.summary.scalar('n_mf_updates', n_mf_updates)

    def _make_tf_model(self):
        self._make_constants()
        self._make_placeholders()
        self._make_vars()
        self._make_train_op()

    def _make_tf_feed_dict(self, X_batch, training=False):
        d = {}
        d['X_batch'] = X_batch
        d['v_rand'] = self._v_layer.make_rand(self.n_particles, self._rng)
        d['h_rand'] = self._h_layers[0].make_rand(self.n_particles, self._rng)
        for i in xrange(1, self.n_layers):
            d['h_rand_{0}'.format(i)] = self._h_layers[i].make_rand(self.n_particles, self._rng)
        if training:
            d['learning_rate'] = self.learning_rate
            d['momentum'] = self.momentum
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

    def _fit(self, X, X_val=None):
        # init generators
        self._learning_rate_gen = make_inf_generator(self.learning_rate)
        self._momentum_gen = make_inf_generator(self.momentum)

        # load ops requested
        self._train_op = tf.get_collection('train_op')[0]
        self._msre = tf.get_collection('msre')[0]
        self._n_mf_updates = tf.get_collection('n_mf_updates')[0]

        # main loop
        for self.epoch in epoch_iter(start_epoch=self.epoch, max_epoch=self.max_epoch,
                                     verbose=self.verbose):
            train_msre, train_n_mf_updates = self._train_epoch(X)

            # print progress
            if self.verbose:
                s = "epoch: {0:{1}}/{2}".format(self.epoch, len(str(self.max_epoch)), self.max_epoch)
                if train_msre:
                    s += "; msre: {0:.5f}".format(train_msre)
                if train_n_mf_updates:
                    s += "; n_mf_upds: {0:.2f}".format(train_n_mf_updates)
                print_inline(s + '\n')

            # save if needed
            if self.save_after_each_epoch:
                self._save_model(global_step=self.epoch)


    @run_in_tf_session
    def gibbs(self, n_steps=5):
        pass


if __name__ == '__main__':
    from rbm import BernoulliRBM
    from hdp_dbm.utils.dataset import load_mnist
    X, _ = load_mnist('train', '../data/')
    X /= 255.
    X = X[:1000]
    #
    # [1]
    rbm1 = BernoulliRBM(n_visible=784,
                        n_hidden=5, # 500
                        n_gibbs_steps=1,
                        w_std=0.001,
                        hb_init=0.,
                        vb_init=0.,
                        learning_rate=0.05,
                        momentum=[.5] * 5 + [.9],
                        batch_size=100,
                        max_epoch=5,#100,
                        L2=1e-3,
                        sample_h_states=True,
                        sample_v_states=True,
                        metrics_config={}, # TODO
                        # verbose=True,
                        random_seed=1337,
                        model_path='dbm-rbm-1/')
    rbm1.fit(X)
    H = rbm1.transform(X)
    # #
    # [2]
    rbm2 = BernoulliRBM(n_visible=5, # 500
                        n_hidden=10, # 1000
                        n_gibbs_steps=2,# [e/20 + 1 for e in xrange(200)],
                        w_std=0.01,
                        hb_init=0.,
                        vb_init=0.,
                        learning_rate=0.05,
                        momentum=[.5] * 5 + [.9],
                        batch_size=100,
                        max_epoch=5,# 200,
                        L2=1e-3,
                        sample_h_states=True,
                        sample_v_states=True,
                        metrics_config={}, # TODO
                        # verbose=True,
                        random_seed=1337,
                        model_path='dbm-rbm-2/')
    rbm2.fit(H)
    Z = rbm2.transform(H)

    rbm3 = BernoulliRBM(n_visible=10,  # 500
                        n_hidden=20,  # 1000
                        n_gibbs_steps=2,  # [e/20 + 1 for e in xrange(200)],
                        w_std=0.01,
                        hb_init=0.,
                        vb_init=0.,
                        learning_rate=0.05,
                        momentum=[.5] * 5 + [.9],
                        batch_size=100,
                        max_epoch=5,  # 200,
                        L2=1e-3,
                        sample_h_states=True,
                        sample_v_states=True,
                        metrics_config={},  # TODO
                        # verbose=True,
                        random_seed=1337,
                        model_path='dbm-rbm-3/')
    rbm3.fit(Z)

    rbm1 = BernoulliRBM.load_model('dbm-rbm-1/')
    print rbm1.get_tf_params(scope='weights')['W'][0][0]
    rbm2 = BernoulliRBM.load_model('dbm-rbm-2/')
    print rbm2.get_tf_params(scope='weights')['W'][0][0]
    rbm3 = BernoulliRBM.load_model('dbm-rbm-3/')
    print rbm3.get_tf_params(scope='weights')['W'][0][0]

    dbm = DBM(rbms=[rbm1, rbm2, rbm3],
              n_particles=10,
              n_gibbs_steps=1, # or 5
              max_mf_updates_per_epoch=10, # or 30
              learning_rate=0.001, # 0.001 -> epsilonw = max(epsilonw/1.000015,0.00010);
              # OR in paper 0.005 + gradually -> 0
              # momentum=???
              max_epoch=100, # 300 or 500 for paper results
              batch_size=100,
              L2=2e-4,
              random_seed=1337,
              # verbose=True,
              save_after_each_epoch=True,
              model_path = 'dbm/')
    dbm.fit(X)
