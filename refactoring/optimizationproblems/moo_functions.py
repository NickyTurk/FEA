from pymoo.factory import get_problem, get_reference_directions, get_visualization
import numpy as np
import autograd.numpy as anp
import math

from refactoring.optimizationproblems.prescription import Prescription


class MOOFunctions:
    def __init__(self, n_dimensions, n_objectives, h, lbound=-100, ubound=100, name='dtlz1'):
        self.problem = None
        self.lbound = lbound
        self.ubound = ubound
        self.function_name = name + '_obj_func'
        self.dim = n_dimensions
        self.n_obj = n_objectives
        self.pareto_front = []
        self.reference_directions = []
        self.h = h  # how many partitions to split each objective in, also used to calculate reference points
        self.population_size = math.comb(h + n_objectives - 1, n_objectives - 1)
        self.k = self.dim - self.n_obj + 1
        self.objectives = []
        getattr(self, name)()

    def run(self, solution, i):
        if np.isinf(i):
            i = self.n_obj
        return getattr(self, self.function_name)(solution, i)

    def g1(self, X_M):
        return 100 * (self.k + anp.sum(anp.square(X_M - 0.5) - anp.cos(20 * anp.pi * (X_M - 0.5))))

    def g2(self, X_M):
        return anp.sum(anp.square(X_M - 0.5), axis=1)

    def dtlz1(self):
        print('entered')
        self.problem = get_problem('dtlz1', n_var=self.dim, n_obj=self.n_obj)
        self.reference_directions = get_reference_directions("das-dennis", self.n_obj, n_partitions=12)
        self.pareto_front = self.problem._calc_pareto_front(self.reference_directions)

    def dtlz1_obj_func(self, x, i):
        X_, X_M = x[:self.n_obj - 1], x[self.n_obj - 1:]
        g = self.g1(X_M)
        if i < self.n_obj:
            _f = 0.5 * (1 + g)
            _f *= np.prod(X_[:X_.shape[0] - i])
            if i > 0:
                _f *= 1 - X_[X_.shape[0] - i]
            return _f
        elif i == self.n_obj:
            f = []
            for j in range(0, self.n_obj):
                _f = 0.5 * (1 + g)
                _f *= np.prod(X_[:X_.shape[0] - j])
                if j > 0:
                    _f *= 1 - X_[X_.shape[0] - j]
                f.append(_f)
            return anp.column_stack(f)


