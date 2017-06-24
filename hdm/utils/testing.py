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
