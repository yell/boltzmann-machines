from copy import deepcopy

from mixin import SeedMixin


def is_param_name(name):
    return not name.startswith('_') and not name.endswith('_')


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
        return params

    def _deserialize(self, params):
        """Class-specific parameters deserialization routine."""
        return params
