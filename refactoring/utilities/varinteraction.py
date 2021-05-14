import numpy as np
from minepy import MINE
from networkx import from_numpy_array, maximum_spanning_tree, connected_components
from random import choice


class MEE(object):
    def __init__(self, func, dim, samples, mic_thresh, de_thresh, delta, use_mic_value=True):
        self.f = func
        self.d = dim
        self.ub = np.ones(self.d) * func.ubound
        self.lb = np.ones(self.d) * func.lbound
        self.samples = samples
        self.mic_thresh = mic_thresh  # mic threshold
        self.de_thresh = de_thresh  # diff equation (de) threshold
        self.delta = delta  # account for small variations
        self.IM = np.zeros((self.d, self.d))
        self.use_mic_value = use_mic_value

    def get_IM(self):
        self.direct_IM()
        if not self.use_mic_value:
            self.strongly_connected_comps()
        return self.IM

    def direct_IM(self):

        """
        Calculates the Direct Interaction Matrix based on MIC
        :return: direct_IM
        """
        f, dim, lb, ub, sample_size, delta = self.f, self.d, self.lb, self.ub, self.samples, self.delta
        # for each dimension
        for i in range(dim):
            # compare to consecutive variable (/dimension)
            for j in range(i + 1, dim):
                # number of values to calculate == sample size
                de = np.zeros(sample_size)
                # generate n values (i.e. samples) for j-th dimension
                x_j = np.random.rand(sample_size) * (ub[j] - lb[j]) + lb[j]
                for k in range(1, sample_size):
                    # randomly generate solution -- initialization of function variables
                    x = np.random.uniform(lb, ub, size=dim)
                    x[j] = x_j[k]  # set jth value to random sample value
                    y_1 = f.run(x)
                    x[i] = x[i] + delta
                    y_2 = f.run(x)
                    de[k] = (y_2 - y_1) / delta

                avg_de = np.mean(de)
                de[de < self.de_thresh] = avg_de  # use np fancy indexing to replace values

                mine = MINE()
                mine.compute_score(de, x_j)
                mic = mine.mic()
                if self.use_mic_value:
                    self.IM[i, j] = mic
                elif not self.use_mic_value and mic > self.mic_thresh:  # threshold <--------
                    self.IM[i, j] = 1
                    self.IM[j, i] = 1

    def strongly_connected_comps(self):
        from networkx import to_networkx_graph, DiGraph
        from networkx.algorithms.components import strongly_connected_components

        """
        Sets strongly connected components in the Interaction Matrix
        """
        IM_graph = to_networkx_graph(self.IM, create_using= DiGraph)
        strongly_connected_components = strongly_connected_components(IM_graph)
        for component in strongly_connected_components:
            component = list(component)
            for i in range(len(component)):
                for j in range(i + 1, len(component)):
                    self.IM[component[i], component[j]] = 1
                    self.IM[component[j], component[i]] = 1


class RandomTree(object):
    def __init__(self, func, dim, samples, de_thresh, delta):
        self.f = func
        self.d = dim
        self.ub = np.ones(self.d) * func.ubound
        self.lb = np.ones(self.d) * func.lbound
        self.delta = delta  # account for small variations
        self.IM = np.ones((self.d, self.d)) * -1  # init IM to bunch of -1's (so we can initialize a tree)
        self.samples = samples
        self.de_thresh = de_thresh

        self.iteration_ctr = 0

        # Init tree and graph
        self.G = from_numpy_array(self.IM)  # We don't technically need this in self, but might as well have it
        self.T = maximum_spanning_tree(self.G)  # just make a tree (they're all -1 so it is a boring tree)

    def run(self, trials):
        for i in range(trials):
            self.iteration_ctr += 1  # keep track of global counter to allow for multiple, sequential run calls
            print("Iteration " + str(self.iteration_ctr))

            edges = list(self.T.edges(data="weight"))
            remove = min(edges)  # find the cheapest edge
            self.T.remove_edge(remove[0], remove[1])  # delete the edge

            comp1, comp2 = connected_components(self.T)

            node1 = choice(list(comp1))  # generate random start node
            node2 = choice(list(comp2))  # generate random end node

            interact = self.compute_interaction(node1, node2)
            if interact > remove[2]:  # if the new random edge is more expensive then the previous one, add it
                self.T.add_edge(node1, node2, weight=interact)
            else:  # otherwise add the original one back
                self.T.add_edge(remove[0], remove[1], weight=remove[2])
        return self.IM

    def compute_interaction(self, i, j):
        if self.IM[i][j] != -1:
            return self.IM[i][j]
        # number of values to calculate == sample size
        f, dim, lb, ub, sample_size, delta = self.f, self.d, self.lb, self.ub, self.samples, self.delta
        de = np.zeros(sample_size)
        # generate n values (i.e. samples) for j-th dimension
        x_j = np.random.rand(sample_size) * (ub[j] - lb[j]) + lb[j]
        for k in range(1, sample_size):
            # randomly generate solution -- initialization of function variables
            x = np.random.uniform(lb, ub, size=dim)
            x[j] = x_j[k]  # set jth value to random sample value
            y_1 = f.run(x)
            x[i] = x[i] + delta
            y_2 = f.run(x)
            de[k] = (y_2 - y_1) / delta

        avg_de = np.mean(de)
        de[de < self.de_thresh] = avg_de  # use np fancy indexing to replace values

        mine = MINE()
        mine.compute_score(de, x_j)
        mic = mine.mic()
        self.IM[i, j] = mic
        return mic


if __name__ == '__main__':
    from refactoring.optimizationProblems.function import Function
    f = Function(function_number=1, shift_data_file="f01_o.txt")
    mee = MEE(f, 5, 5, 0.1, 0.0001, 0.000001)
    mee.get_IM()