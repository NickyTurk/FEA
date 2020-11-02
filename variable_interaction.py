"""
Creation of the variable interaction matrix, to be used for the graph clustering techniques.
VI method TBD, mutual information?
"""

import numpy as np
import copy
from minepy import MINE
from deap.benchmarks import *

class MEE:
    """

    """
    def __init__(self, f, d, ub, lb, n, a, b, delta):
        self.f = f
        self.d = d
        self.ub = ub
        self.lb = lb
        self.n = n
        self.a = a
        self.b = b
        self.delta = delta

    def direct_IM(self):
        """
        Fill out the Interaction Matrix (IM) with directly interacting variables as per
        Sun, Kirley, & Halgamuge (2017). Quantifying  variable  interactions  in  continuous optimization problems.
        :return: direct_IM
        """
        # initialize IM dxd
        IM = np.zeros((self.d, self.d))

        # for each dimension
        for i in range(self.d):
            for j in range(i+1, self.d):
                de = np.zeros(self.n)
                # randomly generate feature values
                # initialization of function variables
                x_0 = np.random.rand(self.d)*(self.ub-self.lb) + self.lb
                # generate n values for j-th dimension
                x_j = np.random.rand(self.n)*(self.ub[j]-self.lb[j]) + self.lb[j]
                for k in range(1, self.n):
                    x = copy.deepcopy(x_0)
                    x[j] = x_j[k]
                    y_1 = self.f(x)
                    x[i] = x[i] + self.delta
                    y_2 = self.f(x)
                    print(y_1)
                    print(y_2)
                    de[k] = (y_2[0] - y_1[0])/self.delta

                avg_de = np.mean(de)

                for k in range(1, self.n):
                    if abs(de[i] - avg_de) < self.b:
                        de[i] = avg_de

                mine = MINE()
                mine.compute_score(de, x_j)
                mic = mine.mic()
                if mic < self.a:
                    IM[i,j] = 1
                    IM[j,i] = 1
        self.direct = IM
        return IM


if __name__ == '__main__':

    mee = MEE(sphere, 3, np.ones(3)*100, np.ones(3)*-100, 50, 0.2, 0.001, 0.000001)
    mee.direct_IM()
    print(mee.direct)
    # def f(x):
    #     return deap.benchmarks.sphere()