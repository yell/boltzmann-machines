from nose.tools import eq_

from boltzmann_machines.base import TensorFlowModel as TFM


class TestWorkingPaths(object):
    def __init__(self):
        pass

    def test_filename_only(self):
        tf_model = TFM(model_path='model')
        eq_(tf_model._model_dirpath, './')
        eq_(tf_model._model_filepath, './model')
        eq_(tf_model._params_filepath, './params.json')
        eq_(tf_model._random_state_filepath, './random_state.json')
        eq_(tf_model._train_summary_dirpath, './logs/train')
        eq_(tf_model._val_summary_dirpath, './logs/val')
        eq_(tf_model._tf_meta_graph_filepath, './model.meta')

        tf_model = TFM(model_path='model-1')
        eq_(tf_model._model_dirpath, './')
        eq_(tf_model._model_filepath, './model-1')
        eq_(tf_model._params_filepath, './params.json')
        eq_(tf_model._random_state_filepath, './random_state.json')
        eq_(tf_model._train_summary_dirpath, './logs/train')
        eq_(tf_model._val_summary_dirpath, './logs/val')
        eq_(tf_model._tf_meta_graph_filepath, './model-1.meta')

    def test_dirname_only(self):
        tf_model = TFM(model_path='a/')
        eq_(tf_model._model_dirpath, 'a/')
        eq_(tf_model._model_filepath, 'a/model')
        eq_(tf_model._params_filepath, 'a/params.json')
        eq_(tf_model._random_state_filepath, 'a/random_state.json')
        eq_(tf_model._train_summary_dirpath, 'a/logs/train')
        eq_(tf_model._val_summary_dirpath, 'a/logs/val')
        eq_(tf_model._tf_meta_graph_filepath, 'a/model.meta')

        tf_model = TFM(model_path='./')
        eq_(tf_model._model_dirpath, './')
        eq_(tf_model._model_filepath, './model')
        eq_(tf_model._params_filepath, './params.json')
        eq_(tf_model._random_state_filepath, './random_state.json')
        eq_(tf_model._train_summary_dirpath, './logs/train')
        eq_(tf_model._val_summary_dirpath, './logs/val')
        eq_(tf_model._tf_meta_graph_filepath, './model.meta')

        tf_model = TFM(model_path='b/a/')
        eq_(tf_model._model_dirpath, 'b/a/')
        eq_(tf_model._model_filepath, 'b/a/model')
        eq_(tf_model._params_filepath, 'b/a/params.json')
        eq_(tf_model._random_state_filepath, 'b/a/random_state.json')
        eq_(tf_model._train_summary_dirpath, 'b/a/logs/train')
        eq_(tf_model._val_summary_dirpath, 'b/a/logs/val')
        eq_(tf_model._tf_meta_graph_filepath, 'b/a/model.meta')

    def test_nothing(self):
        tf_model = TFM(model_path='')
        eq_(tf_model._model_dirpath, './')
        eq_(tf_model._model_filepath, './model')
        eq_(tf_model._params_filepath, './params.json')
        eq_(tf_model._random_state_filepath, './random_state.json')
        eq_(tf_model._train_summary_dirpath, './logs/train')
        eq_(tf_model._val_summary_dirpath, './logs/val')
        eq_(tf_model._tf_meta_graph_filepath, './model.meta')

    def test_all(self):
        tf_model = TFM(model_path='a/b')
        eq_(tf_model._model_dirpath, 'a/')
        eq_(tf_model._model_filepath, 'a/b')
        eq_(tf_model._params_filepath, 'a/params.json')
        eq_(tf_model._random_state_filepath, 'a/random_state.json')
        eq_(tf_model._train_summary_dirpath, 'a/logs/train')
        eq_(tf_model._val_summary_dirpath, 'a/logs/val')
        eq_(tf_model._tf_meta_graph_filepath, 'a/b.meta')

        tf_model = TFM(model_path='./b')
        eq_(tf_model._model_dirpath, './')
        eq_(tf_model._model_filepath, './b')
        eq_(tf_model._params_filepath, './params.json')
        eq_(tf_model._random_state_filepath, './random_state.json')
        eq_(tf_model._train_summary_dirpath, './logs/train')
        eq_(tf_model._val_summary_dirpath, './logs/val')
        eq_(tf_model._tf_meta_graph_filepath, './b.meta')

        tf_model = TFM(model_path='a/b/c')
        eq_(tf_model._model_dirpath, 'a/b/')
        eq_(tf_model._model_filepath, 'a/b/c')
        eq_(tf_model._params_filepath, 'a/b/params.json')
        eq_(tf_model._random_state_filepath, 'a/b/random_state.json')
        eq_(tf_model._train_summary_dirpath, 'a/b/logs/train')
        eq_(tf_model._val_summary_dirpath, 'a/b/logs/val')
        eq_(tf_model._tf_meta_graph_filepath, 'a/b/c.meta')
