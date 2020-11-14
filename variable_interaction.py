"""
Creation of the variable interaction matrix, to be used for the graph clustering techniques.
VI method TBD, mutual information?
"""

import numpy as np
import copy
from minepy import MINE
from deap.benchmarks import *
from cec2013lsgo.cec2013 import Benchmark
from networkx.convert_matrix import *
import networkx as nx

class MEE:
    def __init__(self, f, d, ub, lb, n, a, b, delta):
        self.f = f
        print(self.f)
        self.d = d
        self.ub = ub
        self.lb = lb
        self.n = n
        self.a = a
        self.b = b
        self.delta = delta
        self.IM = np.zeros((self.d, self.d))

    def direct_IM(self):
        """
        algorithm outline
        :return: direct_IM
        """

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
                    print(y_1)
                    x[i] = x[i] + self.delta
                    y_2 = self.f(x)
                    # print(y_1)
                    # print(y_2)
                    if(type(y_2) == type(0.1)):
                        de[k] = (y_2 - y_1)/self.delta
                    else:
                        de[k] = (y_2[0] - y_1[0])/self.delta

                avg_de = np.mean(de)

                for k in range(1, self.n):
                    if abs(de[i] - avg_de) < self.b:
                        de[i] = avg_de

                mine = MINE()
                mine.compute_score(de, x_j)
                mic = mine.mic()
                if mic > self.a:
                    self.IM[i,j] = 1
                    self.IM[j,i] = 1

    def strongly_connected_comps(self):
        IM_graph = nx.to_networkx_graph(self.IM, create_using=nx.DiGraph)
        strongly_connected_components = nx.algorithms.components.strongly_connected_components(IM_graph)
        for component in strongly_connected_components:
            component = list(component)
            for i in range(len(component)):
                for j in range(i+1, len(component)):
                    self.IM[component[i],component[j]] = 1
                    self.IM[component[j],component[i]] = 1

        print(self.IM)




if __name__ == '__main__':
    d = 10
    ubounds = np.ones(d)*100
    lbounds = np.ones(d)*-100
    ubounds[3] *= 0.1
    ubounds[4] *= 0.1
    lbounds[3] *= 0.1
    lbounds[4] *= 0.1
    bench = Benchmark()
    f = bench.get_function(12)
    mee = MEE(f, d, ubounds, lbounds, 50, 0.1, 0.0001, 0.000001)
    mee.direct_IM()
    print(np.array(mee.IM))
    mee.strongly_connected_comps()