from copy import deepcopy
import threading

fitness_lock = threading.Lock()

def make_shifted_fitness_fn(f, shift):
    def h(xs):
        shifted_xs = [x + shift for x in xs]

        return f(shifted_xs)


# def

def make_factored_fitness_fn(factors, solution, f):
    def h(xs):
        temp = deepcopy(solution)
        for i, x in zip(factors, xs):
            temp[i] = x
        # fitness_lock.acquire()
        ret_me = f(temp)
        # fitness_lock.release()
        return ret_me

    return h


# end def

def make_variable_decoder(factors):
    def d(i):
        return factors.index(i)

    return d
# end def
