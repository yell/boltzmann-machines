import os
import json
import tensorflow as tf
from functools import wraps

from base_model import BaseModel


def is_weight_name(name):
    return not name.startswith('_') and name.endswith('_')


def run_in_tf_session(f):
    """Decorator function that takes care to load appropriate graph/session,
    depending on whether model can be loaded from disk or is just created,
    and to execute `f` inside this session.
    """
    @wraps(f)  # preserve bound method properties
    def wrapped(model, *args, **kwargs):
        tf.reset_default_graph()
        model._tf_graph = tf.get_default_graph()
        if model.called_fit: # model should be loaded from disk
            model._tf_saver = tf.train.import_meta_graph(model._tf_meta_graph_filepath)
            with model._tf_graph.as_default():
                with tf.Session() as model._tf_session:
                    model._tf_saver.restore(model._tf_session, model._model_filepath)
                    model._train_op = tf.get_collection('train_op')[0]
                    res = f(model, *args, **kwargs)
        else:
            with model._tf_graph.as_default():
                with tf.Session() as model._tf_session:
                    model._make_tf_model()
                    model._init_tf_ops()
                    res = f(model, *args, **kwargs)
        return res
    return wrapped


class TensorFlowModel(BaseModel):
    def __init__(self, model_path='tf_model/', **kwargs):
        super(TensorFlowModel, self).__init__(**kwargs)
        self._model_dirpath = None
        self._model_filepath = None
        self._params_filepath = None
        self._random_state_filepath = None
        self._train_summary_dirpath = None
        self._val_summary_dirpath = None
        self._tf_meta_graph_filepath = None
        self.setup_working_paths(model_path)

        self.called_fit = False

        self._tf_graph = tf.Graph()
        self._tf_session = None
        self._tf_saver = None
        self._tf_merged_summaries = None
        self._tf_train_writer = None
        self._tf_val_writer = None

    def setup_working_paths(self, model_path):
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
        self._train_summary_dirpath = os.path.join(self._model_dirpath, 'logs/train')
        self._val_summary_dirpath = os.path.join(self._model_dirpath, 'logs/val')
        self._tf_meta_graph_filepath = self._model_filepath + '.meta'

    def _make_tf_model(self):
        raise NotImplementedError

    def _init_tf_ops(self):
        """Initialize all TF variables, operations etc."""
        init_op = tf.global_variables_initializer()
        self._tf_session.run(init_op)
        self._tf_saver = tf.train.Saver()
        self._tf_merged_summaries = tf.summary.merge_all()
        self._tf_train_writer = tf.summary.FileWriter(self._train_summary_dirpath,
                                                      self._tf_graph)
        self._tf_val_writer = tf.summary.FileWriter(self._val_summary_dirpath,
                                                    self._tf_graph)

    def _save_model(self, json_params=None):
        json_params = json_params or {}
        json_params.setdefault('sort_keys', True)
        json_params.setdefault('indent', 4)

        # (recursively) create all folders needed
        for dirpath in (self._train_summary_dirpath, self._val_summary_dirpath):
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)

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
        self._tf_saver.save(self._tf_session, self._model_filepath)

    @classmethod
    def load_model(cls, model_path):
        model = cls(model_path=model_path)

        # update paths
        model.setup_working_paths(model_path)

        # load params
        with open(model._params_filepath, 'r') as params_file:
            params = json.load(params_file)
        model.set_params(**params)

        # restore random state if needed
        if os.path.isfile(model._random_state_filepath):
            with open(model._random_state_filepath, 'r') as random_state_file:
                random_state = json.load(random_state_file)
            model._rng.set_state(random_state)

        # (tf model will be loaded once any computation will be needed)
        return model

    def _fit(self, X, X_val=None):
        """Class-specific `fit` routine."""
        raise NotImplementedError()

    @run_in_tf_session
    def fit(self, X, X_val=None):
        """Fit the model according to the given training data."""
        self._fit(X, X_val=X_val)
        self.called_fit = True
        self._save_model()
        return self

    @run_in_tf_session
    def get_weights(self):
        """Get weights of the model.

        Returns
        -------
        weights : dict
            Weights of the model in form on numpy arrays.
        """
        weights = {}
        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='weights'):
            weights[var.name.split('/')[-1]] = var.eval()
        return weights


if __name__ == '__main__':
    # run corresponding tests
    import env; from utils.testing import run_tests
    import tests.test_tf_model as t
    run_tests(__file__, t)
