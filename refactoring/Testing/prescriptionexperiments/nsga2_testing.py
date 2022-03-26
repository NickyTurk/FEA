import csv
import os
import pickle
from datetime import timedelta
import time

from refactoring.optimizationproblems.prescription import Prescription
from refactoring.MOO.paretofront import *
from refactoring.basealgorithms.MOO_GA import NSGA2
from refactoring.utilities.field.field_creation import Field

field_names = ['Henrys', 'Sec35Mid', 'Sec35West']
field_1 = pickle.load(open('../../utilities/saved_fields/Henrys.pickle', 'rb'))
field_2 = pickle.load(open('../../utilities/saved_fields/sec35mid.pickle', 'rb'))
field_3 = pickle.load(open('../../utilities/saved_fields/sec35west.pickle', 'rb'))
fields_to_test = [field_1, field_2, field_3]

ga_runs = [200]
population_sizes= [500]

for i,field in enumerate(fields_to_test):
    for pop_size in population_sizes:
        for ga_run in ga_runs:
            start = time.time()
            filename = '../../../FEA/results/FEAMOO/NSGA2_' + field_names[
                i] + '_trial_3_objectives_ga_runs_' + str(ga_run) + '_population_' + str(
                pop_size) + time.strftime('_%d%m%H%M%S') + '.pickle'
            nsga = NSGA2(population_size=pop_size, ga_runs=ga_run)
            nsga.run(field=field)
            end = time.time()
            file = open(filename, "wb")
            pickle.dump(nsga, file)
            elapsed = end - start
            print(
                "NSGA with ga runs %d and population %d took %s" % (ga_run, pop_size, str(timedelta(seconds=elapsed))))