import os
from shutil import rmtree
from numpy.testing import (assert_allclose,
                           assert_almost_equal)

from hdp_dbm.utils import RNG
from hdp_dbm.rbm import BaseRBM


class TestBaseRBM(object):
    def __init__(self):
        self.X = RNG(seed=1337).rand(32, 24)
        self.X_val = RNG(seed=42).rand(16, 24)
        self.rbm_config = dict(n_visible=24,
                               n_hidden=16,
                               verbose=False,
                               random_seed=1337)

    def cleanup(self):
        for d in ('test_rbm_1/', 'test_rbm_2/', 'test_rbm_3/'):
            if os.path.exists(d):
                rmtree(d)

    def test_fit_consistency(self):
        # 1) train 3, then 7 more epochs
        rbm = BaseRBM(max_epoch=3,
                      model_path='test_rbm_1/',
                      **self.rbm_config)
        assert_almost_equal(rbm.get_weights()['W:0'][0][0], -0.0094548017)
        rbm.fit(self.X)
        assert_almost_equal(rbm.get_weights()['W:0'][0][0], -0.1969914)
        rbm_weights = rbm.set_params(max_epoch=10) \
            .fit(self.X) \
            .get_weights()
        assert_almost_equal(rbm.get_weights()['W:0'][0][0], -0.22737014)

        # 2) train 3 (+save), load and train 7 more epochs
        rbm2 = BaseRBM(max_epoch=3,
                       model_path='test_rbm_2/',
                       **self.rbm_config)
        rbm2.fit(self.X)
        rbm2_weights = BaseRBM.load_model('test_rbm_2/') \
            .set_params(max_epoch=10) \
            .fit(self.X) \
            .get_weights()
        assert_allclose(rbm_weights['W:0'], rbm2_weights['W:0'])
        assert_allclose(rbm_weights['hb:0'], rbm2_weights['hb:0'])
        assert_allclose(rbm_weights['vb:0'], rbm2_weights['vb:0'])

        # train 10 epochs
        rbm3 = BaseRBM(max_epoch=10,
                       model_path='test_rbm_3/',
                       **self.rbm_config)
        rbm3_weights = rbm3.fit(self.X) \
            .get_weights()
        assert_allclose(rbm2_weights['W:0'], rbm3_weights['W:0'])
        assert_allclose(rbm2_weights['hb:0'], rbm3_weights['hb:0'])
        assert_allclose(rbm2_weights['vb:0'], rbm3_weights['vb:0'])

        # cleanup
        self.cleanup()


    def test_fit_consistency_val(self):
        # 1) train 3, then 7 more epochs
        rbm = BaseRBM(max_epoch=3,
                      model_path='test_rbm_1/',
                      **self.rbm_config)
        assert_almost_equal(rbm.get_weights()['W:0'][0][0], -0.0094548017)
        rbm.fit(self.X, self.X_val)
        assert_almost_equal(rbm.get_weights()['W:0'][0][0], -0.091577843)
        rbm_weights = rbm.set_params(max_epoch=10) \
            .fit(self.X, self.X_val) \
            .get_weights()
        assert_almost_equal(rbm.get_weights()['W:0'][0][0], -0.15510087)

        # 2) train 3 (+save), load and train 7 more epochs
        rbm2 = BaseRBM(max_epoch=3,
                       model_path='test_rbm_2/',
                       **self.rbm_config)
        rbm2.fit(self.X, self.X_val)
        rbm2_weights = BaseRBM.load_model('test_rbm_2/') \
            .set_params(max_epoch=10) \
            .fit(self.X, self.X_val) \
            .get_weights()
        assert_allclose(rbm_weights['W:0'], rbm2_weights['W:0'])
        assert_allclose(rbm_weights['hb:0'], rbm2_weights['hb:0'])
        assert_allclose(rbm_weights['vb:0'], rbm2_weights['vb:0'])

        # train 10 epochs
        rbm3 = BaseRBM(max_epoch=10,
                       model_path='test_rbm_3/',
                       **self.rbm_config)
        rbm3_weights = rbm3.fit(self.X, self.X_val) \
            .get_weights()
        assert_allclose(rbm2_weights['W:0'], rbm3_weights['W:0'])
        assert_allclose(rbm2_weights['hb:0'], rbm3_weights['hb:0'])
        assert_allclose(rbm2_weights['vb:0'], rbm3_weights['vb:0'])

        # cleanup
        self.cleanup()

    def test_transform(self):
        rbm = BaseRBM(max_epoch=3,
                      model_path='test_rbm_1/',
                      **self.rbm_config)
        rbm.fit(self.X)
        H = rbm.transform(self.X_val)

        H_loaded = BaseRBM.load_model('test_rbm_1/').transform(self.X_val)
        assert H.shape == (len(self.X_val), 16)
        assert_allclose(H, H_loaded)

        # cleanup
        self.cleanup()

    def tearDown(self):
        self.cleanup()