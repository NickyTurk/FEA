import csv
import os, re
import pickle
from datetime import timedelta
import time

from refactoring.optimizationproblems.prescription import Prescription
from refactoring.MOO.FEAMOO import FEAMOO
from refactoring.MOO.paretofront import *
from refactoring.basealgorithms.MOO_GA import NSGA2
from refactoring.FEA.factorarchitecture import FactorArchitecture
from refactoring.utilities.field.field_creation import Field

field_names = ['Henrys', 'Sec35West', 'Sec35Mid']
current_working_dir = os.getcwd()
print(current_working_dir)
path = re.search(r'^(.*?[\/\\]FEA)', current_working_dir)
print(path)
path = path.group()
field_1 = pickle.load(open(path + '/refactoring/utilities/saved_fields/Henrys.pickle', 'rb')) # /home/alinck/FEA
field_2 = pickle.load(open(path + '/refactoring/utilities/saved_fields/sec35mid.pickle', 'rb'))
field_3 = pickle.load(open(path + '/refactoring/utilities/saved_fields/sec35west.pickle', 'rb'))
fields_to_test = [field_1, field_2, field_3]

fea_runs = 100
ga_runs = [100]
population_sizes= [500]

def create_strip_groups(field):
    cell_indeces = field.create_strip_trial()
    factors = []
    single_cells = []
    sum = 0
    for j,strip in enumerate(cell_indeces):
        if len(strip.original_index) == 1:
            single_cells.append(sum)
        else:
            factors.append([i+sum for i, og in enumerate(strip.original_index)])
        sum = sum + len(strip.original_index)
    if single_cells:
        factors.append(single_cells)
    print(factors)
    return factors

for i,field in enumerate(fields_to_test):
    print(field_names[i], '-- CCEA')
    FA = FactorArchitecture(len(field.cell_list))
    #FA.linear_grouping(10, 5)
    FA.factors = create_strip_groups(field)
    FA.get_factor_topology_elements()

    ga = NSGA2
    for population in population_sizes:
        for ga_run in ga_runs:
            start = time.time()
            filename = path + '/results/FEAMOO/CCEAMOO_' + field_names[i] + '_trial_3_objectives_strip_topo_ga_runs_' + str(ga_run) + '_population_' + str(population) + time.strftime('_%d%m%H%M%S') + '.pickle'
            feamoo = FEAMOO(Prescription, fea_runs, ga_run, population, FA, ga, dimensions=len(field.cell_list), combinatorial_options=field.nitrogen_list, field=field)
            feamoo.run()
            end = time.time()
            file = open(filename, "wb")
            pickle.dump(feamoo, file)
            elapsed = end-start
            print("FEA with ga runs %d and population %d took %s"%(ga_run, population, str(timedelta(seconds=elapsed))))
