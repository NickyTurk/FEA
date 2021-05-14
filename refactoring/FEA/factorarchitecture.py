import numpy as np
from numba import jit
import scipy as sp

try:
    import _pickle as pickle
except:
    import pickle
import os
from refactoring.utilities.clustering import FuzzyKmeans


@jit
def rotate(xs, n):
    return xs[n:] + xs[:n]


class FactorArchitecture(object):
    """
    Topology Generation:

    With d = 5, there are 5 variables, x_i => x_0, x_1, x_2, x_3, x_4. This means, in any list,
    the variable corresponds to the index. This is a convention and hopefully a space
    saving one.

    Of the four sort of topological elements required by the algorithm, there are two kinds:

    1. Lists where the indices implicitly refer to swarms and the values of the lists
       refer to variables or swarms.
    2. Lists where the indices implicitly refer to variables and the values of the lists
       refer to swarms.

    Factors and neighbors fall into the first group. The factor for swarm 0 is [0, 1].
    The neighbor of swarm 0 is swarm 1 (because they have variables in common).

    Arbiters fall into the second group as do optimizers. The arbiter of variable 0 is swarm 0.
    An arbiter of variable 3 is swarm 3 (0-3). Similarly, optimizers are lists of swarms
    that optimize for a specific index (variable). Variable 1 is optimized by [0, 1] which is
    to say both swarm 0 and swarm 1.

    factors = [[0, 1], [1, 2], [2, 3], [3, 4]] = # of factors and swarms
    neighbors =   [[1], [0, 2], [1, 3], [2]]
    arbiters  =   [0, 1, 2, 3, 3]
    optimizers =  [[0], [0, 1], [1, 2], [2, 3], [3]]
    """

    def __init__(self, dim=0):
        self.factors = []
        self.arbiters = []
        self.optimizers = []
        self.neighbors = []
        self.dim = dim
        self.method = ""
        self.function_evaluations = 0

    def save_architecture(self, path_to_save=""):
        if path_to_save == "":
            if not os.path.isdir("factor_architecture_files/"):
                os.mkdir("factor_architecture_files/")
            if not os.path.isdir("factor_architecture_files/" + self.method):
                os.mkdir("factor_architecture_files/" + self.method)
            file = open("factor_architecture_files/" + self.method + "/" + self.method + "_" + str(self.dim), "wb")
        else:
            folder = os.path.dirname(path_to_save)
            if not os.path.isdir(folder):
                os.mkdir(folder)
            file = open(path_to_save, "wb")
        pickle.dump(self.__dict__, file)

    def load_architecture(self, path_to_load="", method="", dim=0):
        from refactoring.utilities.exceptions import PickleException
        if path_to_load == "" and (method == "" or dim == 0):
            raise PickleException()
        elif path_to_load != "" and os.path.isdir(path_to_load):
            raise PickleException()
        elif path_to_load == "" and method != "" and dim != 0:
            pickle_object = pickle.load(
                open("factor_architecture_files/" + method + "/" + method + "_" + str(dim), 'rb'))
            self.__dict__.update(pickle_object)
        elif path_to_load != "" and not os.path.isdir(path_to_load):
            pickle_object = pickle.load(open(path_to_load, 'rb'))
            self.__dict__.update(pickle_object)

    def load_csv_architecture(self, file, dim, method=""):
        from refactoring.utilities.CSVreader import CSVReader

        csv = CSVReader(file)
        self.factors, f = csv.import_factors(dim)
        self.dim = dim
        self.nominate_arbiters()
        self.calculate_optimizers()
        self.determine_neighbors()

    def linear_grouping(self, width, offset):
        self.method = "linear"
        assert offset <= width
        if offset == width:
            print("WARNING - offset and width are equal; the factors will not overlap.")
        self.factors = list(zip(*[range(i, self.dim, offset) for i in range(0, width)]))
        self.nominate_arbiters()
        self.calculate_optimizers()
        self.determine_neighbors()

    def ring_grouping(self, width=2):
        self.method = "ring"
        self.arbiters = list(range(0, self.dim))
        self.factors = zip(*[rotate(self.arbiters, n) for n in range(0, width)])
        self.determine_neighbors()
        self.calculate_optimizers()

    def diff_grouping(self, _function, epsilon, m=0):
        """
        DIFFERENTIAL GROUPING
        Omidvar et al. 2010
        """
        self.method = "DG"
        size = self.dim
        dimensions = np.arange(start=0, stop=size)
        curr_dim_idx = 0
        factors = []
        separate_variables = []
        function_evaluations = 0
        loop = 0

        while size > 0:
            # initialize for current iteration
            curr_factor = [dimensions[0]]

            curr_factor = self.check_delta(_function, m, 1, size, dimensions, epsilon, curr_factor)

            if len(curr_factor) == 1:
                separate_variables.extend(curr_factor)
            else:
                factors.append(tuple(curr_factor))

            # Final adjustments
            indeces_to_delete = np.searchsorted(dimensions, curr_factor)
            dimensions = np.delete(dimensions, indeces_to_delete)  # remove j from dimensions
            size = len(dimensions)
            if size != 0:
                curr_dim_idx = dimensions[0]

            loop += 1
        if len(separate_variables) != 0:
            factors.append(tuple(separate_variables))

        self.factors = factors
        self.nominate_arbiters()
        self.calculate_optimizers()
        self.determine_neighbors()

    def overlapping_diff_grouping(self, _function, epsilon, m=0):
        """
        Use differential grouping approach to determine factors.
        :return:
        """
        self.method = "ODG"
        size = self.dim
        dimensions = np.arange(start=0, stop=size)
        factors = []
        separate_variables = []
        function_evaluations = 0
        loop = 0

        for i, dim in enumerate(dimensions):
            # initialize for current iteration
            curr_factor = [dim]

            self.check_delta(_function, m, i, size, dimensions, epsilon, curr_factor)

            if len(curr_factor) == 1:
                separate_variables.extend(curr_factor)
            else:
                factors.append(tuple(curr_factor))

            loop += 1

        factors.append(tuple(separate_variables))
        self.factors = factors
        self.nominate_arbiters()
        self.calculate_optimizers()
        self.determine_neighbors()

    def check_delta(self, _function, m, i, size, dimensions, eps, curr_factor):
        """
        Helper function for the two differential grouping approaches.
        Compares function fitnesses to determine whether there is a difference in results larger than 'epsilon'.
        :param _function:
        :param m:
        :param i:
        :param size:
        :param dimensions:
        :param eps:
        :param curr_factor:
        :return curr_factor:
        """
        p1 = np.multiply(_function.lbound, np.ones(self.dim))  # python does weird things if you set p2 = p1
        p2 = np.multiply(_function.lbound, np.ones(self.dim))  # python does weird things if you set p2 = p1
        p2[i] = _function.ubound
        if m == 0:
            delta1 = _function.run(p1) - _function.run(p2)
        else:
            delta1 = _function.run(p1, m_group=m) - _function.run(p2, m_group=m)
        self.function_evaluations += 2

        for j in range(i + 1, size):
            p3 = np.multiply(_function.lbound, np.ones(self.dim))
            p4 = np.multiply(_function.lbound, np.ones(self.dim))
            p4[i] = _function.ubound
            p3[dimensions[j]] = 0
            p4[dimensions[j]] = 0  # grabs dimension to compare to, same as index

            if m == 0:
                delta2 = _function.run(p3) - _function.run(p4)
            else:
                delta2 = _function.run(p3, m_group=m) - _function.run(p4, m_group=m)
            self.function_evaluations += 2

            if abs(delta1 - delta2) > eps:
                curr_factor.append(dimensions[j])

        return curr_factor

    def spectral_grouping(self, IM, num_clusters):
        from networkx import to_networkx_graph, Graph
        from networkx.linalg import laplacian_matrix
        '''
        Assign the datapoints to clusters using spectral clustering and return and array of cluster assignemnts
        '''
        self.method = "spectral"
        IM_graph = to_networkx_graph(IM, create_using=Graph)

        # get Laplacian
        laplacian = sp.sparse.csr_matrix.toarray(laplacian_matrix(IM_graph))

        # calc eigen vectors and values of the laplacian
        eig_values, eig_vectors = np.linalg.eig(laplacian)
        sorted_indices = eig_values.argsort()
        eig_values = eig_values[sorted_indices]
        eig_vectors = eig_vectors[sorted_indices]

        # take k largest eigen vectors
        k_arr = np.arange(num_clusters)
        eig_values = eig_values[k_arr]
        eig_vectors = np.transpose(eig_vectors[k_arr])

        # run fuzzy kmeans with the eigen vectors
        self.factors = FuzzyKmeans(eig_vectors, num_clusters).assign_clusters()
        self.nominate_arbiters()
        self.calculate_optimizers()
        self.determine_neighbors()

    def MEET(self, IM):
        from networkx import  from_numpy_array, maximum_spanning_tree

        """
        Create directed graph with edge weights in MIC table.
        Directed graph (IM) can be calculated using different methods, called from variableinteraction class
        Create MAXimal spanning tree from this graph.
        :return:
        """
        self.method = "MEET"
        G = from_numpy_array(IM)
        T = maximum_spanning_tree(G)

        factors = []

        for node in list(T.nodes):  # each dimension
            factor = list(T.neighbors(node))  # adjacent nodes
            factor.append(node)  # add itself to the group
            factors.append(factor)

        self.factors = factors
        self.nominate_arbiters()
        self.calculate_optimizers()
        self.determine_neighbors()

    def nominate_arbiters(self):
        """
        The arbiter of variable 0 is swarm 0.
        An arbiter of variable 3 is swarm 3 (0-3).
        :return:
        """
        assignments = {}
        # Iteration is faster when it does not have to access the object each time
        factors = self.factors
        for i, factor in enumerate(factors[:-1]):
            for j in factor:
                if j not in self.factors[i + 1] and j not in assignments:
                    assignments[j] = i
        for j in factors[-1]:
            if j not in assignments:
                assignments[j] = len(factors) - 1
        keys = list(assignments.keys())
        keys.sort()
        arbiters = [assignments[k] for k in keys]
        self.arbiters = arbiters

    def calculate_optimizers(self):
        """
        Optimizers are lists of swarms that optimize for a specific index (variable).
        Variable 1 is optimized by swarms [0, 1].
        :return:
        """
        optimizers = []
        factors = self.factors
        for v in range(self.dim):
            optimizer = []
            for i, factor in enumerate(factors):
                if v in factor:
                    optimizer.append(i)
            optimizers.append(optimizer)
        self.optimizers = optimizers

    def determine_neighbors(self):
        """
        The factor for swarm 0 is [0, 1].
        The neighbor of swarm 0 is swarm 1 (because they have variables in common).
        :return:
        """
        neighbors = []
        factors = self.factors
        for i, factor in enumerate(factors):
            neighbor = []
            for j, other_factor in enumerate(factors):
                if (i != j) and not set(factor).isdisjoint(set(other_factor)):
                    neighbor.append(j)
            neighbors.append(neighbor)
        self.neighbors = neighbors
