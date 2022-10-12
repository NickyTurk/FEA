import random
import numpy as np
import scipy as sp

try:
    import _pickle as pickle
except:
    import pickle
import os
# from utilities.clustering import FuzzyKmeans


def rotate(xs, n):
    return xs[n:] + xs[:n]


class FactorArchitecture(object):
    """
    Topology Generation:

    With d = 5, there are 5 variables, x_i => x_0, x_1, x_2, x_3, x_4. This means, in any list,
    the variable corresponds to the home. This is a convention and hopefully a space
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
    that optimize for a specific home (variable). Variable 1 is optimized by [0, 1] which is
    to say both swarm 0 and swarm 1.

    factors = [[0, 1], [1, 2], [2, 3], [3, 4]] = # of factors and swarms
    neighbors =   [[1], [0, 2], [1, 3], [2]]
    arbiters  =   [0, 1, 2, 3, 3]
    optimizers =  [[0], [0, 1], [1, 2], [2, 3], [3]]
    """

    def __init__(self, dim=0, factors=None):
        if factors is not None:
            self.factors = factors
        else:
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
        from utilities.exceptions import PickleException
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
        from utilities.multifilereader import MultiFileReader

        csv = MultiFileReader(file)
        self.factors, f = csv.import_factors(dim)
        self.dim = dim
        self.get_factor_topology_elements()

    def linear_grouping(self, width, offset):
        self.method = "linear"
        assert offset <= width
        if offset == width:
            print("WARNING - offset and width are equal; the factors will not overlap.")
        self.factors = list(zip(*[range(i, self.dim, offset) for i in range(0, width)]))
        if self.factors[-1][-1] != self.dim-1:
            step_back = width - offset
            new_group = tuple(range(self.factors[-1][-step_back], self.dim, 1))
            self.factors.append(new_group)

    def ring_grouping(self, width=2):
        self.method = "ring"
        self.arbiters = list(range(0, self.dim))
        self.factors = zip(*[rotate(self.arbiters, n) for n in range(0, width)])
        self.determine_neighbors()
        self.calculate_optimizers()

    def random_grouping(self, min_groups=5, max_groups=15, overlap=False):
        self.method = "random"
        number_of_groups = random.randint(min_groups,max_groups)
        while True:
            assigned_groups = random.choices(range(0,number_of_groups), k=self.dim)
            if len(set(assigned_groups)) == number_of_groups:
                break
        self.factors = []
        for i in range(number_of_groups):
            factor = [idx for idx, grp in enumerate(assigned_groups) if grp == i]
            self.factors.append(factor)
        if overlap:
            for i in range(0,number_of_groups-1):
                factor = random.sample(self.factors[i],k=int(np.ceil(len(self.factors[i])/4)))
                factor.extend(random.sample(self.factors[i+1],k=int(np.ceil(len(self.factors[i+1])/4))))
                self.factors.append(factor)

    def diff_grouping(self, _function, epsilon, m=0, moo=False, n_obj=np.inf):
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

            curr_factor = self.check_delta(_function, m, 1, size, dimensions, epsilon, curr_factor, moo, n_obj)

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

    def overlapping_diff_grouping(self, _function, epsilon, m=0, moo=False, n_obj=np.inf):
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

            self.check_delta(_function, m, i, size, dimensions, epsilon, curr_factor, moo, n_obj)

            if len(curr_factor) == 1:
                separate_variables.extend(curr_factor)
            else:
                factors.append(tuple(curr_factor))

            loop += 1

        factors.append(tuple(separate_variables))
        self.factors = factors

    def check_delta(self, _function, m, i, size, dimensions, eps, curr_factor, moo=False, n_obj=np.inf):
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
        p1 = [random.random() for x in range(self.dim)]
        p2 = [x for x in p1]
        p2[i] = p1[i]+.1
        if not moo:
            if m == 0:
                delta1 = _function.run(p1) - _function.run(p2)
            else:
                delta1 = _function.run(p1, m_group=m) - _function.run(p2, m_group=m)
        elif moo:
            delta1 = _function.evaluate(p1)[n_obj] - _function.evaluate(p2)[n_obj]
            #print("obj: ", n_obj, "D1: ", delta1)
        self.function_evaluations += 2

        for j in range(i + 1, size):
            p3 = [x for x in p1]
            p4 = [x for x in p2]
            rand = random.random()
            p3[j] = rand
            p4[j] = rand

            if not moo:
                if m == 0:
                    delta2 = _function.run(p3) - _function.run(p4)
                else:
                    delta2 = _function.run(p3, m_group=m) - _function.run(p4, m_group=m)
            elif moo:
                delta2 = _function.evaluate(p3)[n_obj] - _function.evaluate(p4)[n_obj]
                #print("obj: ", n_obj, "D2: ", delta2, _function.evaluate(p3)[n_obj], _function.evaluate(p4)[n_obj])
            self.function_evaluations += 2

            if abs(delta1 - delta2) > eps:
                curr_factor.append(dimensions[j])

        return curr_factor

    # def spectral_grouping(self, IM, num_clusters):
    #     from networkx import to_networkx_graph, Graph
    #     from networkx.linalg import laplacian_matrix
    #     '''
    #     Assign the datapoints to clusters using spectral clustering and return and array of cluster assignemnts
    #     '''
    #     self.method = "spectral"
    #     IM_graph = to_networkx_graph(IM, create_using=Graph)
    #
    #     # get Laplacian
    #     laplacian = sp.sparse.csr_matrix.toarray(laplacian_matrix(IM_graph))
    #
    #     # calc eigen vectors and values of the laplacian
    #     eig_values, eig_vectors = np.linalg.eig(laplacian)
    #     sorted_indices = eig_values.argsort()
    #     eig_values = eig_values[sorted_indices]
    #     eig_vectors = eig_vectors[sorted_indices]
    #
    #     # take k largest eigen vectors
    #     k_arr = np.arange(num_clusters)
    #     eig_values = eig_values[k_arr]
    #     eig_vectors = np.transpose(eig_vectors[k_arr])
    #
    #     # run fuzzy kmeans with the eigen vectors
    #     self.factors = FuzzyKmeans(eig_vectors, num_clusters).assign_clusters()

    def MEET(self, T):
        """
        Create directed graph with edge weights in MIC table.
        Directed graph (IM) can be calculated using different methods, called from variableinteraction class
        Create MAXimal spanning tree from this graph.
        :param T, T is either a directed graph stored as a numpy array, or a networkx graph object
        :return:
        """

        if isinstance(T, np.ndarray):  # convert np array to tree
            from networkx import  from_numpy_array, maximum_spanning_tree
            G = from_numpy_array(T)
            T = maximum_spanning_tree(G)

        self.method = "MEET"
        factors = []

        print(f'Total weight: {T.size(weight="weight")}')

        for node in list(T.nodes):  # each dimension
            factor = list(T.neighbors(node))  # adjacent nodes
            factor.append(node)  # add itself to the group
            factors.append(factor)

        self.factors = factors

    def MEET2(self, T, number_of_factors=50):
        """
        Create directed graph with edge weights in MIC table.
        Directed graph (IM) can be calculated using different methods, called from variableinteraction class
        Create MAXimal spanning tree from this graph.
        :param T, T is either a directed graph stored as a numpy array, or a networkx graph object
        :return:
        """

        if isinstance(T, np.ndarray):  # convert np array to tree
            from networkx import  from_numpy_array, maximum_spanning_tree
            G = from_numpy_array(T)
            T = maximum_spanning_tree(G)

        self.method = "MEET"
        factors = []

        print(f'Total weight: {T.size(weight="weight")}')

        for node in list(T.nodes):  # each dimension
            factor = list(T.neighbors(node))  # adjacent nodes
            factor.append(node)  # add itself to the group
            factors.append(factor)

        while len(factors) > number_of_factors:  # shrink the number of factors!
            factors.sort(key=len)
            f1 = factors.pop(0)
            f2 = factors.pop(0)
            new_f = set(f1 + f2)
            factors.append(list(new_f))

        self.factors = factors

    def get_factor_topology_elements(self):
        """
        Calculates arbiters, optimizers and neighbours based on the created factors.
        """
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
        factors = [f for f in self.factors]
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
        Optimizers are lists of swarms that optimize for a specific home (variable).
        Variable 1 is optimized by swarms [0, 1].
        :return:
        """
        optimizers = []
        factors = [f for f in self.factors]
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
        factors = [f for f in self.factors]
        for i, factor in enumerate(factors):
            neighbor = []
            for j, other_factor in enumerate(factors):
                if (i != j) and not set(factor).isdisjoint(set(other_factor)):
                    neighbor.append(j)
            neighbors.append(neighbor)
        self.neighbors = neighbors


class MooFactorArchitecture:

    def __init__(self, dim, n_obj, problem=None, decomp_approach='diff_grouping'):
        self.dim = dim
        self.problem = problem
        self.n_obj = n_obj
        self.decomp = decomp_approach
        self.n_obj = n_obj

    def graph_based_MOO_dg(self, ubound=1, lbound=0, shift_param=111.1, nr_samples=20):
        # Property analysis
        # Interaction learning
        # Graph grouping
        pass

    def variable_property_analysis(self, shift_param=111.1, nr_samples=20):
        for i in range(self.dim):
            solutions = []
            for j in range(nr_samples):
                shift_value = j-1 + (1/shift_param)
                for k in range(self.dim):
                    if i == k:
                        y = []

    def create_objective_factors(self, save_files=False) -> FactorArchitecture:
        """Create factors along different objective functions.
        For each objective, a FactorArchitecture object is created.
        :param save_files: Boolean that determines whether the created factorArchitectures are saved in pickle files
        :returns FactorArchitecture object: with all the factors generated
        """
        all_factors = FactorArchitecture(self.dim)
        all_factors.method = self.decomp + '_MOO'
        eps = 5
        for i in range(self.n_obj):
            fa = FactorArchitecture(self.dim)
            if self.decomp == 'diff_grouping':
                getattr(fa, self.decomp)(self.problem, eps, moo=True, n_obj=i)
            if save_files:
                fa.save_architecture(
                    '../factor_architecture_files/n_obj_' + str(self.n_obj) + '_' + fa.method + '_dim_' + str(self.dim) + '_obj_' + str(i))
            print(fa.factors)
            print(len(fa.factors))
            for f in fa.factors:
                all_factors.factors.append(f)
        all_factors.get_factor_topology_elements()
        return all_factors

    def read_objective_factors(self, method_name) -> FactorArchitecture:
        all_factors = FactorArchitecture(self.dim)
        for i, obj in enumerate(self.n_obj):
            fa = FactorArchitecture(self.dim)
            fa.load_architecture(
                '../factor_architecture_files/n_obj_' + str(self.n_obj) + '_' + method_name + '_dim_' + str(self.dim) + '_obj_' + str(i))
            all_factors.factors.extend(fa.factors)
        all_factors.get_factor_topology_elements()
        return all_factors


if __name__ == "__main__":
    from pymoo.factory import get_problem
    import re

    current_working_dir = os.getcwd()
    path = re.search(r'^(.*?[\\/]FEA)', current_working_dir)
    path = path.group()

    # dtlz = get_problem("dtlz1", n_var=1000, n_obj=3)
    # moofa = MooFactorArchitecture(dim=1000, problem=dtlz, n_obj=3)
    # factors = moofa.create_objective_factors()
    # print(len(factors.factors))
    # factors.save_architecture(path_to_save=path+"/FEA/factor_architecture_files/DG_MOO/DG_DTLZ1_3_eps5")