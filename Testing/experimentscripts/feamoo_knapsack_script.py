"""
Running Knapsack experiments.
"""

import os, re
import pickle
from datetime import timedelta
import time

from optimizationproblems.knapsack import *
from MOO.MOFEA import MOFEA
from MOO.MOEA import NSGA2, SPEA2, MOEA, MOEAD
from FEA.factorarchitecture import FactorArchitecture
from utilities.util import *


"""
Parameters
"""
nr_items = 1000  # knapsack size
sizes = [100, 100]  # subpopulation size; how many variables are being updated in each subpopulation (100 out of 1000)
overlaps = [80, 100]  # overlap size (here, 80 means 20 overlapping variables, 100 means NO overlap)
fea_runs = [20]  # how many times to run FEA iteration: update, compete, share
ga_run = 100  # how many times to run the base-algorithm for each subpopulation
population = 500  # number of individuals in each subpopulation
ks_type = 'multi'  # type of knapsack: single (balanced) or multi (classic)
nr_obj = 3

# get working directory path to save files
current_working_dir = os.getcwd()
path = re.search(r'^(.*?[\\/]FEA)', current_working_dir)
path = path.group()

"""
Initialize different components
"""
FA = FactorArchitecture(nr_items)
ga = SPEA2  # base algorithm choices: NSGA2, MOEAD, SPEA2
ks = Knapsack(number_of_items=nr_items, max_nr_items=nr_items, nr_objectives=nr_obj, nr_constraints=1,
              knapsack_type=ks_type)  # the knapsack problem


"""
Add fitness evaluation to the base algorithm.
"""
@add_method(MOEA)  # a decorator to assign the fitness evaluation to the correct function
def calc_fitness(variables, gs=None, factor=None):
    # when using FEA/CCEA, we need to get the full solution for which to calculate the fitness
    if gs is not None and factor is not None:
        full_solution = [x for x in gs.variables]
        for i, x in zip(factor, variables):
            full_solution[i] = x
    else:
        full_solution = variables
    # this is the actual fitness calculation for the relevant knapsack
    if ks_type == 'single':
        ks.set_fitness_single_knapsack(full_solution)
    elif ks_type == 'multi':
        ks.set_fitness_multi_knapsack(full_solution)
    return ks.objective_values


"""
The actual experiments
"""
for s, o in zip(sizes, overlaps):  # this groups the subpopulation size with the overlap size
    if s == o:
        name = 'CCSPEA2'
    else:
        name = 'FSPEA2'
    # get the factor architecture based on the size and overlap
    FA.linear_grouping(s, o)
    FA.get_factor_topology_elements()
    for i in range(5):
        print('##############################################\n', i)
        for fea_run in fea_runs:
            start = time.time()
            filename = path + '/results/Knapsack/' + name + '/' + name + '_' + ks_type + '_knapsack_' + str(nr_obj) + \
                       '_objectives_fea_runs_' + str(fea_run) + '_grouping_' + str(s) + '_' + \
                       str(o) + time.strftime('_%d%m%H%M%S') + '.pickle'
            # Initialize MOFEA with factor architecture, base algorithm to use, dimension of problem,
            # combinatorial or continuous variables, and reference point if known.
            feamoo = MOFEA(fea_run, factor_architecture=FA, base_alg=ga, dimensions=nr_items,
                           combinatorial_options=[0, 1], ref_point=ks.ref_point)
            # run the algorithm
            feamoo.run()
            end = time.time()
            # save the results as the entire object
            file = open(filename, "wb")
            pickle.dump(feamoo, file)
            elapsed = end - start
            print(
                "FEA with ga runs %d and population %d %d took %s" % (fea_run, s, o, str(timedelta(seconds=elapsed))))
