import numpy as np
from copy import deepcopy
from ..base.base import is_param_name, is_attribute_name
from .mixin import SeedMixin
from ..utils.utils import write_during_training


class BaseModel(SeedMixin):
    def __init__(self, *args, **kwargs):
        super(BaseModel, self).__init__(*args, **kwargs)

    def get_params(self, deep=True, include_attributes=True):
        """Get parameters (and attributes) of the model.

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
        p = lambda k: is_param_name(k) or (include_attributes and is_attribute_name(k))
        params = {k: params[k] for k in params if p(k)}
        if deep:
            params = deepcopy(params)
        return params

    def set_params(self, **params):
        """Set parameters (and attributes) of the model.

        Parameters
        ----------
        params : kwargs
            Parameters names and their new values.

        Returns
        -------
        self
        """
        for k, v in params.items():
            if (is_param_name(k) or is_attribute_name(k)) and hasattr(self, k):
                setattr(self, k, v)
            else:
                raise ValueError("invalid param name '{0}'".format(k))
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
