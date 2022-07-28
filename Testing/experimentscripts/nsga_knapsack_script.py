"""
Running Knapsack experiments.
"""

import os, re
import pickle
from datetime import timedelta
import time

from optimizationproblems.knapsack import *
from MOO.MOEA import NSGA2
from utilities.util import *

nr_items = 1000
ga_run = 100
population = [500]
objectives = [3, 5]
ks_type = ['multi']

ga = NSGA2

current_working_dir = os.getcwd()
path = re.search(r'^(.*?\/FEA)', current_working_dir)
path = '../../..'

for type in ks_type:
    for obj in objectives:
        ks = Knapsack(number_of_items=nr_items, max_nr_items=nr_items, nr_objectives=obj, nr_constraints=1,
                      knapsack_type=type)

        @add_method(NSGA2)
        def calc_fitness(variables, gs=None, factor=None):
            if gs is not None and factor is not None:
                full_solution = [x for x in gs.variables]
                for i, x in zip(factor, variables):
                    full_solution[i] = x
            else:
                full_solution = variables
            if type == 'single':
                ks.set_fitness_single_knapsack(full_solution)
            elif type == 'multi':
                ks.set_fitness_multi_knapsack(full_solution)
            return ks.objective_values

        for pop in population:
            for i in range(10):
                print('##############################################\n', i)
                start = time.time()
                filename = path + '/results/Knapsack/NSGA2/CNSGA2_' + type + '_knapsack_' + str(obj) + '_objectives_ga_runs_' + str(
                ga_run) + '_population_' + str(pop) + '_' + time.strftime('_%d%m%H%M%S') + '.pickle'
                nsga = NSGA2(dimensions=nr_items, population_size=pop, ea_runs=ga_run, combinatorial_values=[0, 1],
                             ref_point=ks.ref_point)
                nsga.run()
                end = time.time()
                file = open(filename, "wb")
                pickle.dump(nsga, file)
                elapsed = end - start
                print(
                    "NSGA with ga runs %d and population %d took %s" % (
                    ga_run, pop, str(timedelta(seconds=elapsed))))