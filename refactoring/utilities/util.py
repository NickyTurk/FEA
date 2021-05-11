from functools import wraps


def memo(f):
    cache = {}

    @wraps(f)
    def wrap(*arg):
        if arg not in cache:
            cache['arg'] = f(*arg)
            return cache['arg']
    return wrap
