import numpy as np
from copy import deepcopy

from base import is_param_name
from mixin import SeedMixin
from bm.utils import write_during_training


class BaseModel(SeedMixin):
    def __init__(self, *args, **kwargs):
        super(BaseModel, self).__init__(*args, **kwargs)

    def get_params(self, deep=True):
        """Get parameters of the model.

        Parameters
        ----------
        deep : bool, optional
            Whether to deepcopy all the parameters.

        Returns
        -------
        params : dict
            Parameters of the model.
        """
        params = vars(self)
        params = {key: params[key] for key in params if is_param_name(key)}
        if deep:
            params = deepcopy(params)
        return params

    def set_params(self, **params):
        """Set parameters of the model.

        Parameters
        ----------
        params : kwargs
            Parameters names and their new values.

        Returns
        -------
        self
        """
        for key, value in params.items():
            if is_param_name(key) and hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError("invalid param name '{0}'".format(key))
        return self

    def _serialize(self, params):
        """Class-specific parameters serialization routine."""
        for k, v in params.items():
            if isinstance(v, np.ndarray):
                if v.size > 1e6:
                    msg = "WARNING: parameter `{0}` won't be serialized because it is too large:"
                    msg += ' ({1:.2f} > 1 Mio elements)'
                    msg = msg.format(k, 1e-6 * v.size)
                    write_during_training(msg)
                    params[k] = None
                else:
                    params[k] = v.tolist()
        return params

    def _deserialize(self, params):
        """Class-specific parameters deserialization routine."""
        return params
