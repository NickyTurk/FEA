"""
Creation of the variable interaction matrix, to be used for the graph clustering techniques.
VI method TBD, mutual information?
"""

import numpy as np
from numpy import linalg as la
import math
import copy
from minepy import MINE
from deap.benchmarks import *
# from cec2013lsgo.cec2013 import Benchmark
from networkx.convert_matrix import *
import networkx as nx


class MEE:
    def __init__(self, f, d, ub, lb, n, mic_thresh, de_thresh, delta):
        self.f = f
        print(self.f)
        self.d = d
        self.ub = ub
        self.lb = lb
        self.n = n
        self.mic_thresh = mic_thresh  # mic threshold
        self.de_thresh = de_thresh  # diff equation (de) threshold
        self.delta = delta  # account for small variations
        self.IM = np.zeros((self.d, self.d))

    def direct_IM(self):

        """
            Calculates teh Direct Interaction Matrix based on MIC
            :return: direct_IM
            """

        # for each dimension
        for i in range(self.d):
            for j in range(i + 1, self.d):
                de = np.zeros(self.n)
                # randomly generate feature values -- initialization of function variables
                x_0 = np.random.rand(self.d) * (self.ub - self.lb) + self.lb
                # generate n values for j-th dimension
                x_j = np.random.rand(self.n) * (self.ub[j] - self.lb[j]) + self.lb[j]
                for k in range(1, self.n):
                    x = copy.deepcopy(x_0)
                    x[j] = x_j[k]
                    y_1 = self.f(x)
                    x[i] = x[i] + self.delta
                    y_2 = self.f(x)
                    de[k] = (y_2 - y_1) / self.delta

                avg_de = np.mean(de)

                for k in range(1, self.n):
                    if abs(de[i] - avg_de) < self.de_thresh:
                        de[i] = avg_de

                mine = MINE()
                mine.compute_score(de, x_j)
                mic = mine.mic()
                if mic > self.mic_thresh:  # threshold <--------
                    self.IM[i, j] = 1
                    self.IM[j, i] = 1

    def strongly_connected_comps(self):
        """
        Sets strongly connected components in the Interaction Matrix
        """
        IM_graph = nx.to_networkx_graph(self.IM, create_using=nx.DiGraph)
        strongly_connected_components = nx.algorithms.components.strongly_connected_components(IM_graph)
        for component in strongly_connected_components:
            component = list(component)
            for i in range(len(component)):
                for j in range(i + 1, len(component)):
                    self.IM[component[i], component[j]] = 1
                    self.IM[component[j], component[i]] = 1


class MEE_groups:
    def __init__(self, f, d, ub, lb, n, direct_thresh, indirect_thresh, de_thresh, delta):
        self.f = f
        print(self.f)
        self.d = d
        self.ub = ub
        self.lb = lb
        self.n = n
        self.direct_thresh = direct_thresh  # direct interaction threshold
        self.indirect_thresh = indirect_thresh  # indirect interaction threshold (overlapping groups)
        self.de_thresh = de_thresh  # diff equation (de) threshold
        self.delta = delta  # account for small variations
        self.direct_IM = np.zeros((self.d, self.d))
        self.indirect_IM = np.zeros((self.d, self.d))
        self.mic = np.zeros((self.d, self.d))


    def create_groups(self):
        """
        Calculates the Direct Interaction Matrix based on MIC
        """

        # for each dimension
        for i in range(self.d):
            for j in range(i + 1, self.d):
                de = np.zeros(self.n)
                # randomly generate feature values -- initialization of function variables
                x_0 = np.random.rand(self.d) * (self.ub - self.lb) + self.lb
                # generate n values for j-th dimension
                x_j = np.random.rand(self.n) * (self.ub[j] - self.lb[j]) + self.lb[j]
                for k in range(1, self.n):
                    x = copy.deepcopy(x_0)
                    x[j] = x_j[k]
                    y_1 = self.f(x)
                    x[i] = x[i] + self.delta
                    y_2 = self.f(x)
                    de[k] = (y_2 - y_1) / self.delta

                avg_de = np.mean(de)

                for k in range(1, self.n):
                    if abs(de[i] - avg_de) < self.de_thresh:
                        de[i] = avg_de

                mine = MINE()
                mine.compute_score(de, x_j)
                mic = mine.mic()
                self.mic[i, j] = mic
                if mic > self.direct_thresh:  # direct threshold should make several disconnected components in graph
                    self.direct_IM[i, j] = 1
                    self.direct_IM[j, i] = 1

                if mic > self.indirect_thresh:  # indirect threshold. Should add overlap to disconnected components
                    self.indirect_IM[i, j] = 1
                    self.indirect_IM[j, i] = 1

        self.dynamic_thresh(2)  # reassigns dynamic threshold so that graph has 2 groups
        self.assign_groups()

        return self.groups

    def dynamic_thresh(self, num_disjoint):
        best_thresh = 0
        best_groups = 1
        best_IM = np.zeros((self.d, self.d))
        for thresh in np.linspace(0, 1, 21):  # should make points separated by 0.05
            print(thresh)
            IM = np.zeros((self.d, self.d))
            disjoint = 0
            for i in range(self.d):
                for j in range(i + 1, self.d):
                    if self.mic[i, j] > thresh:  # direct threshold should make several disconnected components in graph
                        IM[i, j] = 1
                        IM[j, i] = 1

            direct_graph = nx.to_networkx_graph(IM, create_using=nx.DiGraph)
            disjoint_groups = nx.algorithms.components.strongly_connected_components(direct_graph)
            nodes_in_disjoint = set()
            for g in disjoint_groups:
                if len(g) > 1:
                    disjoint += 1
                    nodes_in_disjoint = nodes_in_disjoint.union(set(g))

            if num_disjoint - disjoint <= num_disjoint - best_groups:  # select by closest to target number of groups
                # could down-select here for fewer nodes in the disjoint (more overlap) or opposite
                best_groups = disjoint
                best_thresh = thresh
                best_IM = IM

        print('return')
        self.direct_IM = best_IM
        return IM

    def assign_groups(self):
        # Groups have been made. We can use direct_IM as adjacency graph with disconnected components
        direct_graph = nx.to_networkx_graph(self.direct_IM, create_using=nx.DiGraph)
        disjoint_groups = nx.algorithms.components.strongly_connected_components(direct_graph)
        nodes_in_disjoint = set()
        disjoint_list = []
        for g in disjoint_groups:
            disjoint_list.append(g)
            if len(g) > 1:
                nodes_in_disjoint = nodes_in_disjoint.union(g)
            else:
                print('Disjoint Dim: ' + str(list(g)))
        # nodes_in_disjoint = set([dim for component in disjoint_groups for dim in component])  # flattens disjoint_groups then makes set of it

        indirect_graph = nx.to_networkx_graph(self.indirect_IM, create_using=nx.DiGraph)
        components = nx.algorithms.components.strongly_connected_components(indirect_graph)

        self.groups = []

        for c in components:
            for g in disjoint_list:
                print(c)
                print(g)
                set_g = set(g)
                set_c = set(c)
                if len(set_c.intersection(set_g)) > 0:  # shares nodes so combine
                    overlapping_dims = set_c - nodes_in_disjoint
                    group = set_g.union(overlapping_dims)
                    self.groups.append(group)




class CMA:
    # f: function optimizing
    # dim: number dimentions (n)
    # k: number individuals each generation
    # u: number individuals to keep each gen
    def __init__(self, f, dim):
        self.dim = dim
        self.f = f
        pass

    # Copied from Matlab implementation in (Completely Derandomized Self-Adaptation in Evolution Strategies)
    # Lamb ~ 4 + floor(3log(N)), mu ~ floor(lamb/2)
    def cmaes(self, lamb, mu):
        N = self.dim

        # Stopping criteria
        maxeval = 300 * (N + 2) ^ 2
        stopfit = 10 ** -10

        xmeanw = np.ones((N, 1))  # Object parameter start (weighted mean)

        # Step size
        sigma = 1.0
        minsigma = 10 ^ -15

        arweights = np.log((lamb + 1) / 2) - np.log([i for i in range(1, mu + 1)])  # for recombination

        # Adaptation
        cc = 4 / (N + 4)
        ccov = 2 / ((N + np.sqrt(2)) ** 2)
        cs = cc
        damp = 1 / cs + 1

        # Dynamic
        # TODO: fix B,D so they are the bounds of the function
        B = np.identity(N)
        D = np.identity(N)
        BD = np.matmul(B, D)
        C = np.matmul(BD, np.transpose(BD))
        pc = np.zeros((N, 1))
        ps = np.copy(pc)
        cw = sum(arweights) / la.norm(arweights)
        chiN = np.sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * (N ** 2)))

        # Generation loop
        counteval = 0
        arfitness = np.zeros(lamb)
        arfitness[0] = 2 * abs(stopfit)
        arz = []
        arx = []
        tups = []
        while arfitness[0] > stopfit & counteval < maxeval:
            # Generate lambda offspring
            for k in range(lamb):
                # TODO, make sure x inside bounds
                arz.append(np.random.randn(N))
                x = xmeanw + sigma * (np.matmul(BD, arz[k]))
                arx.append(x)
                arfitness[k] = self.f(x)
            tups = [(i, arx[i]) for i in range(len(arx))]
            tups = sorted(tups, key=lambda t: t[1], reverse=True)

            zmeanw = np.ones((N, 1))
            for i in range(len(xmeanw)):
                xmeanw = arx[tups[i][0]] * arweights[i]
                zmeanw = arz[tups[i][0]] * arweights[i]

            # Adapt covariance
            pc = (1 - cc) * pc + (np.sqrt(cc * (2 - cc)) * cw) * (np.matmul(BD, zmeanw))
            C = (1 - ccov) * C + ccov * np.matmul(pc, np.transpose(pc))

            # adapt sigma
            ps = (1 - cs) * ps + (sqrt(cs * (2 - cs)) * cw) * (np.matmul(B, zmeanw))
            sigma = sigma * np.exp((la.norm(ps) - chiN) / chiN / damp)

            if counteval / lamb % N / 10 < 1:
                C = np.triu(C) + np.transpose(np.triu(C, 1))
                D, B = la.eig(C)
                D = np.diag(np.sqrt(np.diag(D)))
                BD = np.matmul(B, D)

        return arx[tups[0][0]]


def cigar(x):
    f = x[0] ** 2
    b = 10 ** 6
    for i in range(1, len(x)):
        f += b * x[i] ** 2
    return f


if __name__ == '__main__':
    cmaes = CMA(cigar, 10)
    l = 4 + np.floor(3 * np.log(10))
    u = np.floor(l / 2)
    cmaes.cmaes(l, u)
    # d = 10
    # ubounds = np.ones(d)*100
    # lbounds = np.ones(d)*-100
    # ubounds[3] *= 0.1
    # ubounds[4] *= 0.1
    # lbounds[3] *= 0.1
    # lbounds[4] *= 0.1
    # bench = Benchmark()
    # f = bench.get_function(12)
    # mee = MEE(f, d, ubounds, lbounds, 50, 0.1, 0.0001, 0.000001)
    # mee.direct_IM()
    # print(np.array(mee.IM))
    # mee.strongly_connected_comps()
