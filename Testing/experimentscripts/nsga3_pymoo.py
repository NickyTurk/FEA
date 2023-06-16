import os
import pickle
import re
import time

from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.algorithms.moo import nsga2, nsga3
from pymoo.util.ref_dirs import get_reference_directions

dimensions = 100
ga_run = 100
population = 1000
n_objs = [5, 10]
problems = ['DTLZ5']

current_working_dir = os.getcwd()
path = re.search(r'^(.*?[\\/]FEA)', current_working_dir)
path = path.group()

for i in range(10):
    print("iteration; ", i)
    for problem in problems:
        for n_obj in n_objs:
            print(problem, n_obj)
            function = get_problem(problem, n_var=dimensions, n_obj=n_obj)
            ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=4)
            nsga3obj = nsga3.NSGA3(ref_dirs=ref_dirs, pop_size=population)
            res = minimize(function,
                           nsga3obj,
                           seed=None,
                           termination=('n_gen', ga_run))
            #stats_dict = po.evaluate_solution(res.F, reference_point, normalize=True)
            # print('NSGA3: ', len(res.F))

            filename = path + '/results/factorarchive/'+problem+'/NSGA3/NSGA3_4partitions_'+problem+'_'+str(dimensions)+'_dim_' + str(n_obj) + \
                                             '_objectives_'+str(ga_run)+'_ea_runs_'+str(population)+'_population'+time.strftime('_%d%m%H%M%S')+'.pickle'
            file = open(filename, "wb")
            pickle.dump(res, file)
            file.close()