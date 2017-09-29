import os
import numpy as np
from shutil import rmtree
from numpy.testing import (assert_allclose,
                           assert_almost_equal,
                           assert_raises)

from hdm.utils import RNG
from hdm.rbm import BernoulliRBM, MultinomialRBM, GaussianRBM


class TestRBM(object):
    def __init__(self):
        self.n_visible = 12
        self.n_hidden = 8
        self.X = RNG(seed=1337).rand(16, self.n_visible)
        self.X_val = RNG(seed=42).rand(8, self.n_visible)
        self.rbm_config = dict(n_visible=self.n_visible, n_hidden=self.n_hidden,
                               L2=0., sample_h_states=False,
                               verbose=False, random_seed=1337)

    def cleanup(self):
        for d in ('test_rbm_1/', 'test_rbm_2/', 'test_rbm_3/'):
            if os.path.exists(d):
                rmtree(d)

    def test_w_init(self):
        assert_raises(ValueError, lambda: BernoulliRBM(n_visible=4, n_hidden=3, w_init=np.zeros((4, 2))))
        assert_raises(ValueError, lambda: BernoulliRBM(n_visible=4, n_hidden=3, w_init=np.zeros((3, 3))))
        assert_raises(ValueError, lambda: BernoulliRBM(n_visible=4, n_hidden=3, w_init=np.zeros((3, 2))))
        BernoulliRBM(n_visible=4, n_hidden=3, w_init=np.zeros((4, 3)))
        BernoulliRBM(n_visible=3, n_hidden=3, w_init=np.zeros((3, 3)))
        BernoulliRBM(n_visible=1, n_hidden=1, w_init=np.zeros((1, 1)))

    def test_fit_consistency(self):
        for C, dtype in (
            (BernoulliRBM, 'float32'),
            (BernoulliRBM, 'float64'),
            (MultinomialRBM, 'float32'),
            (GaussianRBM, 'float32'),
        ):
            # 1) train for 2 epochs, then for 3 more
            rbm = C(max_epoch=2,
                    model_path='test_rbm_1/',
                    tf_dtype=dtype,
                    **self.rbm_config)

            if dtype == 'float32': assert_almost_equal(rbm.get_tf_params(scope='weights')['W'][0][0], -0.0094548017)
            if dtype == 'float64': assert_almost_equal(rbm.get_tf_params(scope='weights')['W'][0][0], -0.0077341544416)
            rbm.fit(self.X)
            rbm_weights = rbm.set_params(max_epoch=2 + 3) \
                .fit(self.X) \
                .get_tf_params(scope='weights')

            # 2) train for 2 epochs (+save), load and train for 3 more
            rbm2 = C(max_epoch=2,
                     model_path='test_rbm_2/',
                     tf_dtype=dtype,
                     **self.rbm_config)
            rbm2.fit(self.X)
            rbm2_weights = C.load_model('test_rbm_2/') \
                .set_params(max_epoch=2 + 3) \
                .fit(self.X) \
                .get_tf_params(scope='weights')
            assert_allclose(rbm_weights['W'], rbm2_weights['W'])
            assert_allclose(rbm_weights['hb'], rbm2_weights['hb'])
            assert_allclose(rbm_weights['vb'], rbm2_weights['vb'])

            # train for 5 epochs
            rbm3 = C(max_epoch=2 + 3,
                     model_path='test_rbm_3/',
                     tf_dtype=dtype,
                     **self.rbm_config)
            rbm3_weights = rbm3.fit(self.X) \
                .get_tf_params(scope='weights')
            assert_allclose(rbm2_weights['W'], rbm3_weights['W'])
            assert_allclose(rbm2_weights['hb'], rbm3_weights['hb'])
            assert_allclose(rbm2_weights['vb'], rbm3_weights['vb'])

            # cleanup
            self.cleanup()

    def test_fit_consistency_val(self):
        for C in (BernoulliRBM,):

            # 1) train for 2 epochs, then for 3 more
            rbm = C(max_epoch=2,
                    model_path='test_rbm_1/',
                    **self.rbm_config)

            rbm.fit(self.X, self.X_val)
            rbm_weights = rbm.set_params(max_epoch=2 + 3) \
                .fit(self.X, self.X_val) \
                .get_tf_params(scope='weights')

            # 2) train for 2 epochs (+save), load and train for 3 more
            rbm2 = C(max_epoch=2,
                     model_path='test_rbm_2/',
                     **self.rbm_config)
            rbm2.fit(self.X, self.X_val)
            rbm2_weights = C.load_model('test_rbm_2/') \
                .set_params(max_epoch=2 + 3) \
                .fit(self.X, self.X_val) \
                .get_tf_params(scope='weights')
            assert_allclose(rbm_weights['W'], rbm2_weights['W'])
            assert_allclose(rbm_weights['hb'], rbm2_weights['hb'])
            assert_allclose(rbm_weights['vb'], rbm2_weights['vb'])

            # train for 5 epochs
            rbm3 = C(max_epoch=2 + 3,
                     model_path='test_rbm_3/',
                     **self.rbm_config)
            rbm3_weights = rbm3.fit(self.X, self.X_val) \
                .get_tf_params(scope='weights')
            assert_allclose(rbm2_weights['W'], rbm3_weights['W'])
            assert_allclose(rbm2_weights['hb'], rbm3_weights['hb'])
            assert_allclose(rbm2_weights['vb'], rbm3_weights['vb'])

            # cleanup
            self.cleanup()

    def test_transform(self):
        for C in (BernoulliRBM, MultinomialRBM, GaussianRBM):
            rbm = C(max_epoch=2,
                    model_path='test_rbm_1/',
                    **self.rbm_config)
            rbm.fit(self.X)
            H = rbm.transform(self.X_val)

            H_loaded = C.load_model('test_rbm_1/').transform(self.X_val)
            assert H.shape == (len(self.X_val), self.n_hidden)
            assert_allclose(H, H_loaded)

            # cleanup
            self.cleanup()

    def tearDown(self):
        self.cleanup()
