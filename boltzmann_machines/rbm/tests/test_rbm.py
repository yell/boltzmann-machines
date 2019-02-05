import os
import numpy as np
from shutil import rmtree
from numpy.testing import (assert_allclose,
                           assert_almost_equal,
                           assert_raises)

from rbm import BernoulliRBM, MultinomialRBM, GaussianRBM
from utils import RNG


class TestRBM(object):
    def __init__(self):
        self.n_visible = 12
        self.n_hidden = 8
        self.X = RNG(seed=1337).rand(16, self.n_visible)
        self.X_val = RNG(seed=42).rand(8, self.n_visible)
        self.rbm_config = dict(n_visible=self.n_visible, n_hidden=self.n_hidden,
                               sample_v_states=True, sample_h_states=True,
                               dropout=0.9,
                               verbose=False, display_filters=False,
                               random_seed=1337)

    def cleanup(self):
        for d in ('test_rbm_1/', 'test_rbm_2/'):
            if os.path.exists(d):
                rmtree(d)

    def test_W_init(self):
        for C in (BernoulliRBM, MultinomialRBM, GaussianRBM):
            assert_raises(ValueError, lambda: C(n_visible=4, n_hidden=3, W_init=np.zeros((4, 2))))
            assert_raises(ValueError, lambda: C(n_visible=4, n_hidden=3, W_init=np.zeros((3, 3))))
            assert_raises(ValueError, lambda: C(n_visible=4, n_hidden=3, W_init=np.zeros((3, 2))))
            C(n_visible=4, n_hidden=3, W_init=np.zeros((4, 3)))
            C(n_visible=3, n_hidden=3, W_init=np.zeros((3, 3)))
            C(n_visible=1, n_hidden=1, W_init=np.zeros((1, 1)))

    def compare_weights(self, rbm1, rbm2):
        rbm1_weights = rbm1.get_tf_params(scope='weights')
        rbm2_weights = rbm2.get_tf_params(scope='weights')
        assert_allclose(rbm1_weights['W'], rbm2_weights['W'])
        assert_allclose(rbm1_weights['hb'], rbm2_weights['hb'])
        assert_allclose(rbm1_weights['vb'], rbm2_weights['vb'])

    def compare_transforms(self, rbm1, rbm2):
        H1 = rbm1.transform(self.X_val)
        H2 = rbm2.transform(self.X_val)
        assert H1.shape == (len(self.X_val), self.n_hidden)
        assert H1.shape == H2.shape
        assert_allclose(H1, H2)

    def test_initialization(self):
        for C, dtype in (
                (BernoulliRBM, 'float32'),
                (BernoulliRBM, 'float64'),
                (MultinomialRBM, 'float32'),
                (GaussianRBM, 'float32'),
        ):
            rbm = C(max_epoch=2,
                    model_path='test_rbm_1/',
                    dtype=dtype,
                    **self.rbm_config)
            rbm.init()
            if dtype == 'float32':
                assert_almost_equal(rbm.get_tf_params(scope='weights')['W'][0][0], -0.0094548017)
            if dtype == 'float64':
                assert_almost_equal(rbm.get_tf_params(scope='weights')['W'][0][0], -0.0077341544416)

    def test_consistency(self):
        for C, dtype in (
            (BernoulliRBM, 'float32'),
            (BernoulliRBM, 'float64'),
            (MultinomialRBM, 'float32'),
            (GaussianRBM, 'float32'),
        ):
            # train 2 RBMs with same params for 2 epochs
            rbm1 = C(max_epoch=2,
                     model_path='test_rbm_1/',
                     dtype=dtype,
                     **self.rbm_config)
            rbm2 = C(max_epoch=2,
                     model_path='test_rbm_2/',
                     dtype=dtype,
                     **self.rbm_config)

            rbm1.fit(self.X)
            rbm2.fit(self.X)

            self.compare_weights(rbm1, rbm2)
            self.compare_transforms(rbm1, rbm2)

            # train for 1 more epoch
            rbm1.set_params(max_epoch=rbm1.max_epoch + 1).fit(self.X)
            rbm2.set_params(max_epoch=rbm2.max_epoch + 1).fit(self.X)

            self.compare_weights(rbm1, rbm2)
            self.compare_transforms(rbm1, rbm2)

            # load from disk
            rbm1 = C.load_model('test_rbm_1/')
            rbm2 = C.load_model('test_rbm_2/')

            self.compare_weights(rbm1, rbm2)
            self.compare_transforms(rbm1, rbm2)

            # train for 1 more epoch
            rbm1.set_params(max_epoch=rbm1.max_epoch + 1).fit(self.X)
            rbm2.set_params(max_epoch=rbm2.max_epoch + 1).fit(self.X)

            self.compare_weights(rbm1, rbm2)
            self.compare_transforms(rbm1, rbm2)

            # cleanup
            self.cleanup()

    def test_consistency_val(self):
        rbm1 = BernoulliRBM(max_epoch=2,
                            model_path='test_rbm_1/',
                            **self.rbm_config)
        rbm2 = BernoulliRBM(max_epoch=2,
                            model_path='test_rbm_2/',
                            **self.rbm_config)

        rbm1.fit(self.X, self.X_val)
        rbm2.fit(self.X, self.X_val)

        self.compare_weights(rbm1, rbm2)
        self.compare_transforms(rbm1, rbm2)

        # cleanup
        self.cleanup()

    def tearDown(self):
        self.cleanup()
