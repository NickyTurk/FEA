import numpy as np
from minepy import MINE


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
        import networkx as nx
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


if __name__ == '__main__':
    from refactoring.optimizationProblems.function import Function
    f = Function(function_number=1, shift_data_file="f01_o.txt")
    mee = MEE(f, 5, 5, 0.1, 0.0001, 0.000001)
    mee.get_IM()