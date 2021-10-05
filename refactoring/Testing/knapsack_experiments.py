import os, re
import pickle
from datetime import timedelta
import time

from refactoring.optimizationproblems.knapsack import *
from refactoring.MOO.FEAMOO import FEAMOO
from refactoring.basealgorithms.MOO_GA import NSGA2
from refactoring.FEA.factorarchitecture import FactorArchitecture
from refactoring.utilities.util import *

nr_items = 1000
sizes = [200]  #, 100, 200]c
overlaps = [20]  #, 10, 20]
fea_runs = [20]
ga_run = 100
population = 500

FA = FactorArchitecture(nr_items)

ga = NSGA2
ks = Knapsack(number_of_items=nr_items, max_bag_weight=1600, max_nr_items=nr_items, max_bag_volume=2600,
              nr_objectives=3)

current_working_dir = os.getcwd()
path = re.search(r'^(.*?\/FEA)', current_working_dir)
path = '../..'


@add_method(NSGA2)
def calc_fitness(variables, gs=None, factor=None):
    if gs is not None and factor is not None:
        full_solution = [x for x in gs.variables]
        for i, x in zip(factor, variables):
            full_solution[i] = x
    else:
        full_solution = variables
    ks.set_fitness(full_solution)
    return ks.objective_values

for i in range(1):
    for s, o in zip(sizes, overlaps):
        FA.linear_grouping(s, o)
        # FA.factors = create_strip_groups(field)
        FA.get_factor_topology_elements()
        for fea_run in fea_runs:
                start = time.time()
                filename = path + '/results/Knapsack/FEA/FEA_knapsack_3_objectives_fea_runs_' + str(
                    fea_run) + '_grouping_' + str(s) + '_' + str(o) + time.strftime('_%d%m%H%M%S') + '.pickle'
                feamoo = FEAMOO(fea_run, ga_run, population, FA, ga, dimensions=nr_items,
                                combinatorial_options=[0, 1], ref_point=ks.ref_point)
                feamoo.run()
                #nsga = NSGA2(dimensions=nr_items, population_size=population, ea_runs=ga_run, combinatorial_values=[0, 1],
                #             ref_point=ks.ref_point)
                #nsga.run()
                end = time.time()
                file = open(filename, "wb")
                pickle.dump(feamoo, file)
                elapsed = end - start
                print(
                    "FEA with ga runs %d and population %d %d took %s" % (fea_run, s, o, str(timedelta(seconds=elapsed))))
