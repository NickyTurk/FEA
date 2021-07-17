from functools import wraps
import pyproj, string
from functools import partial


def memo(f):
    cache = {}

    @wraps(f)
    def wrap(*arg):
        if arg not in cache:
            cache['arg'] = f(*arg)
            return cache['arg']
    return wrap

def project():
    return partial(
        pyproj.transform,
        pyproj.Proj('+init=EPSG:26912', preserve_units=True),  # 26912 , 32612
        pyproj.Proj('+init=EPSG:4326'))  # 4326


def project_to_meters(x, y):
    inProj = pyproj.Proj(init='epsg:4326')
    outProj = pyproj.Proj(init='epsg:3857')
    xp, yp = pyproj.transform(inProj, outProj, x, y)
    return xp, yp

def is_hex(s):
    hex_digits = set(string.hexdigits)
    return all(c in hex_digits for c in s)