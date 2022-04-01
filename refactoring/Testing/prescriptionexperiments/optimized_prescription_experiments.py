from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
from refactoring.predictionalgorithms.yieldprediction import YieldPredictor
from refactoring.optimizationproblems.prescription import Prescription
from refactoring.FEA.factorarchitecture import FactorArchitecture
from refactoring.utilities.util import *
from refactoring.basealgorithms.MOO_GA import *
from refactoring.MOO.FEAMOO import FEAMOO
import pandas as pd
from datetime import timedelta
import numpy as np
import pickle, random, re, os, time

fea_runs = 20
ga_runs = [50]
population_sizes = [200]
upper_bound = 150

field_names = ['Sec35Mid', 'Sec35West'] #'Henrys',
current_working_dir = os.getcwd()
path = re.search(r'^(.*?\\FEA)',current_working_dir)
path = path.group()

field_1 = pickle.load(open(path + '/refactoring/utilities/saved_fields/Henrys.pickle', 'rb')) # /home/alinck/FEA
field_2 = pickle.load(open(path + '/refactoring/utilities/saved_fields/sec35mid.pickle', 'rb'))
field_3 = pickle.load(open(path + '/refactoring/utilities/saved_fields/sec35west.pickle', 'rb'))
fields_to_test = [field_2] #[field_1, field_2, field_3]

for i, field in enumerate(fields_to_test):
    agg_files = ["C:/Users/f24n127/Documents/Work/Ag/Data/broyles_sec35mid_2016_yl_aggreg_20181112.csv"]
    df = pd.read_csv(agg_files[i])
    y_labels = df['yl_2016']
    data_to_use = ['x', 'y', 'n_lbs_ac', 'elev_m', 'slope_deg', 'ndvi_2012', 'ndvi_2014', 'ndvi_2015', 'yl14_nn_bu_ac',
                   'n15_lbs_ac', 'n14_lbs_ac']
    x_data = df[data_to_use]
    rf = RandomForestRegressor()
    rf.fit(x_data, y_labels)

    print(field_names[i], '-- NSGA')
    field.fixed_costs = 1000
    #random_global_variables = random.choices([20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150], k=len(field.cell_list))
    random_global_variables = [random.randrange(0, upper_bound) for x in range(len(field.cell_list))]
    pr = Prescription(variables=random_global_variables, field=field)
    yp = YieldPredictor(prescription=pr, field=field, agg_data_file=agg_files[i], trained_model=rf, data_headers=data_to_use)
    FA = FactorArchitecture(len(field.cell_list))
    FA.factors = field.create_strip_groups()
    FA.get_factor_topology_elements()
    nsga = NSGA2

    @add_method(NSGA2)
    def calc_fitness(variables, gs=None, factor=None):
        pres = Prescription(variables=variables, field=field, factor=factor, optimized=True, yield_predictor=yp)
        if gs:
            #global_solution = Prescription(variables=gs.variables, field=field)
            pres.set_fitness(global_solution=gs.variables, cont_bool=True)
        else:
            pres.set_fitness(cont_bool=True)
        return pres.objective_values

    for j in range(5):
        for population in population_sizes:
            for ga_run in ga_runs:
                start = time.time()
                filename = path + '/results/prescriptions/optimized/FEAMOO_' + field_names[i] + '_strip_trial_3_objectives_ga_runs_' + str(ga_run) + '_population_' + str(population) + time.strftime('_%d%m%H%M%S') + '.pickle'
                feamoo = FEAMOO(fea_runs, ga_run, population, FA, nsga, dimensions=len(field.cell_list),
                                upper_value_limit=upper_bound, ref_point=[20000, 1, 1000]) #, combinatorial_options=field.nitrogen_list)
                feamoo.run()
                #nsga = NSGA2(population_size=population, ea_runs=ga_run, dimensions=len(field.cell_list),
                #             upper_value_limit=150, ref_point=[1, 1, 1])
                #nsga.run()
                end = time.time()
                file = open(filename, "wb")
                pickle.dump(feamoo, file)
                elapsed = end-start
                print("NSGA with ga runs %d and population %d took %s"%(ga_run, population, str(timedelta(seconds=elapsed))))
