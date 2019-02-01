import numpy as np
import tensorflow as tf

from ..utils import RNG


class BaseMixin(object):
    def __init__(self, *args, **kwargs):
        if args or kwargs:
            raise AttributeError('Invalid parameters: {0}, {1}'.format(args, kwargs))
        super(BaseMixin, self).__init__()


class DtypeMixin(BaseMixin):
    def __init__(self, dtype='float32', *args, **kwargs):
        super(DtypeMixin, self).__init__(*args, **kwargs)
        self.dtype = dtype

    @property
    def _tf_dtype(self):
        return getattr(tf, self.dtype)

    @property
    def _np_dtype(self):
        return getattr(np, self.dtype)


class SeedMixin(BaseMixin):
    def __init__(self, random_seed=None, *args, **kwargs):
        super(SeedMixin, self).__init__(*args, **kwargs)
        self.random_seed = random_seed
        self._rng = RNG(seed=self.random_seed)

    def make_random_seed(self):
        return self._rng.randint(2 ** 31 - 1)
