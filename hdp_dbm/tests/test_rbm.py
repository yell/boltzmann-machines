from numpy.testing import (assert_allclose,
                           assert_almost_equal)

from hdp_dbm.utils import RNG
from hdp_dbm.rbm import BaseRBM


class TestBaseRBM(object):
    def __init__(self):
        self.X = RNG(seed=1337).rand(32, 24)
        self.rbm_config = dict(n_visible=24,
                               n_hidden=16,
                               verbose=False,
                               random_seed=1337)

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
