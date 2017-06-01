import numpy as np


class RNG(np.random.RandomState):
    """Class encapsulating random number generator.

    Creates RNG from `seed`:
    If `seed` is None, return default RNG.
    If `seed` is int or [int], return new RNG instance seeded with it.

    Raises
    ------
    TypeError
        If `seed` is none from the above.

    Examples
    --------
    >>> rng = RNG(1337)
    >>> rng.rand()
    0.2620246750155817
    >>> rng.rand()
    0.1586839721544656
    >>> rng.reseed()
    >>> rng.rand()
    0.2620246750155817
    >>> rng.rand()
    0.1586839721544656
    """
    def __init__(self, seed=None):
        self._seed = seed
        super(RNG, self).__init__(self._seed)

    def reseed(self):
        if self._seed is not None:
            self.seed(self._seed)


if __name__ == '__main__':
    # run corresponding tests
    from testing import run_tests
    run_tests(__file__)
