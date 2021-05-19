# <codecell>

from collections import namedtuple
import random as rng
from functools import reduce 

class Random:
    def __init__(self):
        self.number_of_calls = 0
    # def
    def uniform( self, low, high):
        self.number_of_calls += 1
        return rng.uniform( low, high)
    # def
    def random(self):
        self.number_of_calls += 1
        return rng.random()
    # def
    def calls(self):
        return self.number_of_calls
    # def
    def reset( self):
        self.number_of_calls = 0
# class

# http://stackoverflow.com/questions/2955412/python-destructuring-bind-dictionary-contents
# If used to retrieve a single value instead of get or [],
# remember that you're using tuple assigment so:
# a, = [1]
# and not:
# a = [1]
pluck = lambda dict, *args: (dict[arg] for arg in args)
# <markdowncell>

# We want versions of the functions `add`, `sub`, and `mul` that can
# handle multiple arguments of mixed types, scalars and lists.
#
# The strategy is to use the basic function versions of these operations
# from the operator module. We then create a factory function that returns
# a version of the argument with the above characteristics.
#
# That function itself is just a reduce over the arguments using
# the binary version of the operator and type checking to make
# the application of the binary version make sense.
#
# <codecell>

import operator as op

# <codecell>
def make_multi_arg_operator(operator):
    def operation(*args):
        def _op( xs, ys):
            if isinstance( xs, list) and isinstance( ys, list):
                return [operator( x, y) for x, y in zip( xs, ys)]
            if isinstance( xs, list):
                return [operator( x, ys) for x in xs]
            if isinstance( ys, list):
                return [operator( xs, y) for y in ys]
            return operator( xs, ys)
        result = reduce(_op, args)
        return result
    return operation

# <markdowncell>

# Now we use `make_multi_arg_operator_func` to create our versions
# of `add`, `sub` and `mul`. Remember the goal here is that we hope
# these versions are fast on PyPy because Numpy is not fully implemented
# for it.

# <codecell>
add = make_multi_arg_operator( op.add)
sub = make_multi_arg_operator( op.sub)
mul = make_multi_arg_operator( op.mul)

# <markdowncell>

# We need a way to merge a dictionary as in Clojure's merge function.

#
#
#

# http://stackoverflow.com/questions/38987/how-to-merge-two-python-dictionaries-in-a-single-expression
def dict_merge(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.'''
    z = x.copy()
    z.update(y)
    return z

def merge_dicts(*dict_args):
    '''
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    '''
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

# <markdowncell>

# We create a named Tuple called `Particle` that holds the position,
# velocity, and fitness of a PSO particle. Personal bests are held
# within the Swarm itself as a separate list.

# <codecell>

Particle = namedtuple( 'Particle', ["position", "velocity", "fitness"])


def pp( xs):
    return map( lambda v: "%.4f" % v, xs)
#
