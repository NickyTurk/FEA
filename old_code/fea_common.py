from copy import deepcopy
import threading
import numpy as np

fitness_lock = threading.Lock()


def make_shifted_fitness_fn(f, shift):
    def h(xs):
        shifted_xs = [x + shift for x in xs]
        return f(np.array(shifted_xs))


def make_factored_fitness_fn(factors, solution, f):
    temp = deepcopy(solution)
    def h(xs):
        for i, x in zip(factors, xs):
            temp[i] = x
        ret_me = f(np.array(temp))
        return ret_me

    return h


def make_variable_decoder(factors):
    def d(i):
        return factors.home(i)

    return d
