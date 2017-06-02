import os
import json
import tensorflow as tf

from base_model import BaseModel


def is_weight_name(name):
    return not name.startswith('_') and name.endswith('_')


class TensorFlowModel(BaseModel):
    def __init__(self, model_path='', save_model=True, **kwargs):
        super(TensorFlowModel, self).__init__(**kwargs)
        self._model_dirpath = None
        self._model_filepath = None
        self._params_filepath = None
        self._random_state_filepath = None
        self._summary_dirpath = None
        self._tf_meta_graph_filepath = None
        self._setup_working_paths(model_path)

        self.save_model = save_model
        self.called_fit = False
        self._tf_graph = tf.Graph()
        self._tf_merged_summaries = None
        self._tf_saver = None
        self._tf_session = None
        self._tf_summary_writer = None

    def _setup_working_paths(self, model_path):
        """
        Parameters
        ----------
        model_path : str
            Model dirpath (should contain slash at the end) or filepath
        """
        head, tail = os.path.split(model_path)
        if not head: head = '.'
        if not tail: tail = 'model'
        self._model_dirpath = head
        if not self._model_dirpath.endswith('/'): self._model_dirpath += '/'
        self._model_filepath = os.path.join(self._model_dirpath, tail)
        self._params_filepath = os.path.join(self._model_dirpath, 'params.json')
        self._random_state_filepath = os.path.join(self._model_dirpath, 'random_state.json')
        self._summary_dirpath = os.path.join(self._model_dirpath, 'logs')
        self._tf_meta_graph_filepath = self._model_filepath + '.meta'

    def _make_tf_model(self):
        raise NotImplementedError

    def _init_tf_ops(self):
        """Initialize all TF variables, operations etc."""
        init_op = tf.global_variables_initializer()
        self._tf_session.run(init_op)
        if self._tf_merged_summaries is None:
            self._tf_merged_summaries = tf.summary.merge_all()
        if self._tf_saver is None:
            self._tf_saver = tf.train.Saver()
        if self._tf_summary_writer is None and self.save_model:
            self._tf_summary_writer = tf.summary.FileWriter(self._summary_dirpath,
                                                            self._tf_session.graph)

    def _save_model(self, json_params=None, tf_save_params=None):
        json_params = json_params or {}
        tf_save_params = tf_save_params or {}
        json_params.setdefault('sort_keys', True)
        json_params.setdefault('indent', 4)

        # (recursively) create all folders needed
        if not os.path.exists(self._summary_dirpath):
            os.makedirs(self._summary_dirpath)

        # save params
        params = self.get_params(deep=False)
        with open(self._params_filepath, 'w') as params_file:
            json.dump(params, params_file, **json_params)

        # dump random state if needed
        if self.random_seed is not None:
            random_state = self._rng.get_state()
            with open(self._random_state_filepath, 'w') as random_state_file:
                json.dump(random_state, random_state_file)

        # save tf model
        with self._tf_graph.as_default():
            self._tf_saver.save(self._tf_session, self._model_filepath, **tf_save_params)

    @classmethod
    def load_model(cls, model_dirpath):
        model = cls(model_dirpath=model_dirpath)

        # update paths
        model._setup_working_paths(model_dirpath)

        # load params
        with open(model._params_filepath, 'r') as params_file:
            params = json.load(params_file)
        model.set_params(**params)

        # restore random state if needed
        if os.path.isfile(model._random_state_filepath):
            with open(model._random_state_filepath, 'r') as random_state_file:
                random_state = json.load(random_state_file)
            model._rng.set_state(random_state)

        # load tf model
        model._tf_saver = tf.train.import_meta_graph(os.path.join(model._model_dirpath, 'model.meta'))
        model._tf_graph = tf.get_default_graph()
        with model._tf_graph.as_default():
            model._make_tf_model()
            # model._tf_saver = tf.train.Saver()
            with tf.Session() as model._tf_session:
                init_op = tf.global_variables_initializer()
                model._tf_session.run(init_op)
                model._tf_saver.restore(model._tf_session, model._model_filepath)
        return model

    def _fit(self, X, *args, **kwargs):
        """Class-specific `fit` routine."""
        raise NotImplementedError()

    def fit(self, X, *args, **kwargs):
        """Fit the model according to the given training data."""
        with self._tf_graph.as_default():
            if not self.called_fit:
                self._make_tf_model()
            with tf.Session() as self._tf_session:
                self._init_tf_ops()
                self._fit(X, *args, **kwargs)
                self.called_fit = True
                if self.save_model:
                    self._save_model()
        return self

    def get_weights(self):
        """Get weights of the model.

        Returns
        -------
        weights : dict
            Weights of the model in form on numpy arrays.
        """
        if not self.called_fit:
            raise ValueError('`fit` must be called before calling `get_weights`')
        if not self.save_model:
            raise RuntimeError('model not found, rerun with `save_model`=True')
        # collect and filter all attributes
        weights = vars(self)
        weights = {key: weights[key] for key in weights if is_weight_name(key)}
        # evaluate the respective variables
        # with self._tf_graph.as_default():
        with tf.Session() as self._tf_session:
            self._tf_saver = tf.train.import_meta_graph(os.path.join(self._model_dirpath, 'model.meta'))
            self._tf_saver.restore(self._tf_session, self._model_filepath)
            for key, value in weights.items():
                weights[key] = value.eval()
        return weights

if __name__ == '__main__':
    # run corresponding tests
    import env; from utils.testing import run_tests
    from tests import test_tf_model
    run_tests(__file__, test_tf_model)
