from datetime import timedelta

from pymoo.decomposition.tchebicheff import Tchebicheff
from pymoo.decomposition.pbi import PBI
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.problems import get_problem

from MOO.MOEA import NSGA2, SPEA2, MOEAD, MOEA
from FEA.factorarchitecture import FactorArchitecture
from MOO.MOFEA import MOFEA
from utilities.util import *

import os, re, time, pickle
from pymoo.problems.many.dtlz import DTLZ1

"""
PARAMETERS
"""
dimensions = 100
s = 10 # , 200, 100]
o = 10  # 100, 160, 80]  # , 10, 20]
fea_runs = [10]
ga_run = 10
population = 100
nr_objs = [3]
problems = ['WFG4', 'WFG5','WFG7'] #DTLZ7 7with PBI for all obj. linear and random
groupings = ["random"]
overlap_bool=True
iter = 5

# pf = get_problem(problem, n_var=dimensions, n_obj=nr_obj).pareto_front(ref_dirs)
# reference_point = np.max(f, axis=0)

# Get path to local working directory
current_working_dir = os.getcwd()
path = re.search(r'^(.*?[\\/]FEA)', current_working_dir)
path = path.group()

"""
Beginning of experiment loops
"""
for nr_obj in nr_objs:
    # Get reference directions for use in MOEA/D
    if nr_obj > 3:
        ref_dirs = get_reference_directions("das-dennis", nr_obj, n_partitions=4)
    else:
        ref_dirs = get_reference_directions("das-dennis", nr_obj, n_partitions=12)

    # Initialize base algorithms for use in MOFEA
    moea1 = partial(SPEA2, population_size=population, ea_runs=ga_run)
    moea2 = partial(NSGA2, population_size=population, ea_runs=ga_run)
    moea3 = partial(MOEAD, ea_runs=ga_run, weight_vector=ref_dirs, n_neighbors=10, problem_decomposition=Tchebicheff()) # PBI(theta=5)

    # Set the base algorithms and their names
    partial_methods = [moea1]
    names=['SPEA2']  # 'SPEA2', 'NSGA2',

    """
    Create appropriate factor architecture for MOFEA
    """
    FA = FactorArchitecture(dimensions)

    for grouping in groupings:
        if grouping == "linear":
            FA.linear_grouping(s, o)
            FA.method = "linear_" + str(s) + '_' + str(o)
        elif grouping == "random":
            FA.classic_random_grouping(100, overlap=overlap_bool)
        FA.get_factor_topology_elements()

        for problem in problems:
            """
            This creates the appropriate fitness function.
            @add_method is a decorator function that allows you to overwrite the fitness function.
            """
            @add_method(MOEA)
            def calc_fitness(variables, gs=None, factor=None):
                if gs is not None and factor is not None:
                    full_solution = np.array([x for x in gs.variables])
                    for i, x in zip(factor, variables):
                        full_solution[i] = x
                else:
                    full_solution = np.array(variables)
                # this is where the actual fitness if calculated
                dtlz = get_problem(problem, n_var=dimensions, n_obj=nr_obj)
                objective_values = dtlz.evaluate(full_solution)
                return tuple(objective_values)

            # Start of iterations
            for i in range(iter):
                for j,alg in enumerate(partial_methods):
                    if not overlap_bool:
                        name = 'CC' + names[j]
                    else:
                        name = 'F' + names[j]
                    print(name, FA.method, problem, str(nr_obj))
                    print('##############################################\n', i)
                    for fea_run in fea_runs:
                        start = time.time()
                        filename = path + '/results/'+problem+'/' + name + '/' + name + '_'+problem+'_' + str(nr_obj) + \
                                '_objectives_fea_runs_' + str(fea_run) + '_grouping_' + FA.method + time.strftime('_%d%m%H%M%S') + '.pickle'
                        feamoo = MOFEA(fea_run, factor_architecture=FA, base_alg=alg, dimensions=dimensions,
                                    value_range=[0.0, 1.0], ref_point=np.ones(nr_obj))
                        feamoo.run()
                        end = time.time()
                        """
                        This is all to save files in the appropriate folders
                        This should honesly really be checked before running the experiments to save time on potential mistakes
                        """
                        try:
                            file = open(filename, "wb")
                            pickle.dump(feamoo, file)
                        except OSError:
                            if not os.path.isdir(path + '/results/' + problem.upper() + '/' + name + '/'):
                                try:
                                    os.mkdir(path + '/results/' + problem.upper() + '/' + name + '/')
                                    file = open(filename, "wb")
                                    pickle.dump(feamoo, file)
                                except OSError:
                                    if not os.path.isdir(path + '/results/' + problem.upper() + '/'):
                                        os.mkdir(path + '/results/' + problem.upper() + '/')
                                        os.mkdir(path + '/results/' + problem.upper() + '/' + name + '/')
                                    file = open(filename, "wb")
                                    pickle.dump(feamoo, file)
                        elapsed = end - start
                        print(
                            "FEA with ga runs %d and population %s took %s" % (fea_run, FA.method, str(timedelta(seconds=elapsed))))
