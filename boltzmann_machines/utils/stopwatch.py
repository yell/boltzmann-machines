import sys
import time


class Stopwatch(object):
    """
    A simple cross-platform
    context-manager stopwatch.

    Examples
    --------
    >>> import time
    >>> with Stopwatch(verbose=True) as s:
    ...     time.sleep(0.1) # doctest: +ELLIPSIS
    Elapsed time: 0.10... sec
    >>> with Stopwatch(verbose=False) as s:
    ...     time.sleep(0.1)
    >>> import math
    >>> math.fabs(s.elapsed() - 0.1) < 0.05
    True
    """
    def __init__(self, verbose=False):
        self.verbose = verbose
        if sys.platform == 'win32':
            # on Windows, the best timer is time.clock()
            self._timer_func = time.clock
        else:
            # on most other platforms, the best timer is time.time()
            self._timer_func = time.time
        self.reset()

    def __enter__(self, verbose=False):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return self.elapsed()

    def start(self):
        if not self._is_running:
            self._start = self._timer_func()
            self._is_running = True
        return self

    def stop(self):
        if self._is_running:
            self._total += (self._timer_func() - self._start)
            self._is_running = False
        return self

    def elapsed(self):
        if self._is_running:
            now = self._timer_func()
            self._total += (now - self._start)
            self._start = now
        if self.verbose:
            print("Elapsed time: {0:.3f} sec".format(self._total))
        return self._total

    def reset(self):
        self._start = 0.
        self._total = 0.
        self._is_running = False
        return self
