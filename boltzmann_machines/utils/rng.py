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
    >>> state = rng.get_state()
    >>> state1 = rng.get_state()
    >>> rng.rand()
    0.2620246750155817
    >>> rng.rand()
    0.1586839721544656
    >>> _ = rng.reseed()
    >>> rng.rand()
    0.2620246750155817
    >>> rng.rand()
    0.1586839721544656
    >>> _ = rng.set_state(state)
    >>> rng.rand()
    0.2620246750155817
    >>> import json
    >>> with open('random_state.json', 'w') as f:
    ...     json.dump(state1, f)
    >>> with open('random_state.json', 'r') as f:
    ...     loaded_state = json.load(f)
    >>> rng.set_state(loaded_state).rand()
    0.2620246750155817
    """
    def __init__(self, seed=None):
        self._seed = seed
        super(RNG, self).__init__(self._seed)

    def reseed(self):
        if self._seed is not None:
            self.seed(self._seed)
        return self

    def get_state(self):
        """Get JSON-serializable inner state."""
        state = super(RNG, self).get_state()
        state = list(state)
        state[1] = state[1].tolist()
        return state

    def set_state(self, state):
        """Complementary method to `get_state`."""
        state[1] = np.asarray(state[1], dtype=np.uint32)
        state = tuple(state)
        super(RNG, self).set_state(state)
        return self


if __name__ == '__main__':
    # run corresponding tests
    from .testing import run_tests
    run_tests(__file__)
