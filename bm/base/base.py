def is_param_name(name):
    return not name.startswith('_') and not name.endswith('_')

def is_attribute_name(name):
    return not name.startswith('_') and name.endswith('_')
