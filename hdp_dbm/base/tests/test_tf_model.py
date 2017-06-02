from hdp_dbm.base import TensorFlowModel as TFM


class TestWorkingPaths(object):
    def __init__(self):
        pass

    def test_filename_only(self):
        tf_model = TFM(model_path='model')
        assert tf_model._model_dirpath == './'
        assert tf_model._model_filepath == './model'
        assert tf_model._params_filepath == './params.json'
        assert tf_model._random_state_filepath == './random_state.json'
        assert tf_model._summary_dirpath == './logs'
        assert tf_model._tf_meta_graph_filepath == './model.meta'

        tf_model = TFM(model_path='model-1')
        assert tf_model._model_dirpath == './'
        assert tf_model._model_filepath == './model-1'
        assert tf_model._params_filepath == './params.json'
        assert tf_model._random_state_filepath == './random_state.json'
        assert tf_model._summary_dirpath == './logs'
        assert tf_model._tf_meta_graph_filepath == './model-1.meta'

    def test_dirname_only(self):
        tf_model = TFM(model_path='a/')
        assert tf_model._model_dirpath == 'a/'
        assert tf_model._model_filepath == 'a/model'
        assert tf_model._params_filepath == 'a/params.json'
        assert tf_model._random_state_filepath == 'a/random_state.json'
        assert tf_model._summary_dirpath == 'a/logs'
        assert tf_model._tf_meta_graph_filepath == 'a/model.meta'

        tf_model = TFM(model_path='./')
        assert tf_model._model_dirpath == './'
        assert tf_model._model_filepath == './model'
        assert tf_model._params_filepath == './params.json'
        assert tf_model._random_state_filepath == './random_state.json'
        assert tf_model._summary_dirpath == './logs'
        assert tf_model._tf_meta_graph_filepath == './model.meta'

        tf_model = TFM(model_path='b/a/')
        assert tf_model._model_dirpath == 'b/a/'
        assert tf_model._model_filepath == 'b/a/model'
        assert tf_model._params_filepath == 'b/a/params.json'
        assert tf_model._random_state_filepath == 'b/a/random_state.json'
        assert tf_model._summary_dirpath == 'b/a/logs'
        assert tf_model._tf_meta_graph_filepath == 'b/a/model.meta'

    def test_nothing(self):
        tf_model = TFM(model_path='')
        assert tf_model._model_dirpath == './'
        assert tf_model._model_filepath == './model'
        assert tf_model._params_filepath == './params.json'
        assert tf_model._random_state_filepath == './random_state.json'
        assert tf_model._summary_dirpath == './logs'
        assert tf_model._tf_meta_graph_filepath == './model.meta'

        tf_model = TFM()
        assert tf_model._model_dirpath == './'
        assert tf_model._model_filepath == './model'
        assert tf_model._params_filepath == './params.json'
        assert tf_model._random_state_filepath == './random_state.json'
        assert tf_model._summary_dirpath == './logs'
        assert tf_model._tf_meta_graph_filepath == './model.meta'

    def test_all(self):
        tf_model = TFM(model_path='a/b')
        assert tf_model._model_dirpath == 'a/'
        assert tf_model._model_filepath == 'a/b'
        assert tf_model._params_filepath == 'a/params.json'
        assert tf_model._random_state_filepath == 'a/random_state.json'
        assert tf_model._summary_dirpath == 'a/logs'
        assert tf_model._tf_meta_graph_filepath == 'a/b.meta'

        tf_model = TFM(model_path='./b')
        assert tf_model._model_dirpath == './'
        assert tf_model._model_filepath == './b'
        assert tf_model._params_filepath == './params.json'
        assert tf_model._random_state_filepath == './random_state.json'
        assert tf_model._summary_dirpath == './logs'
        assert tf_model._tf_meta_graph_filepath == './b.meta'

        tf_model = TFM(model_path='a/b/c')
        assert tf_model._model_dirpath == 'a/b/'
        assert tf_model._model_filepath == 'a/b/c'
        assert tf_model._params_filepath == 'a/b/params.json'
        assert tf_model._random_state_filepath == 'a/b/random_state.json'
        assert tf_model._summary_dirpath == 'a/b/logs'
        assert tf_model._tf_meta_graph_filepath == 'a/b/c.meta'
