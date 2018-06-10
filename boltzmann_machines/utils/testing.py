import nose


@nose.tools.nottest
def run_tests(script_path, test_module=None):
    """
    Run tests which are contained in `test_module` for script
    whose location is specified in `script_path` (typically, is
    called as __file__).
    """
    params = ['', script_path]
    if test_module:
        params.append(test_module.__file__)
    params.append('--with-doctest')
    nose.run(argv=params)

def assert_shape(obj, name, desired_shape):
    actual_shape = getattr(obj, name).shape
    if actual_shape != desired_shape:
        raise ValueError('`{0}` has invalid shape {1} != {2}'.\
                         format(name, actual_shape, desired_shape))

def assert_len(obj, name, desired_len):
    actual_len = len(getattr(obj, name))
    if actual_len != desired_len:
        raise ValueError('`{0}` has invalid len {1} != {2}'.\
                         format(name, actual_len, desired_len))
