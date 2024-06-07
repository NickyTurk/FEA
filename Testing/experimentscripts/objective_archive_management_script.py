import os
import pickle
import re
import time
from datetime import timedelta

from MOO.archivemanagement import ObjectiveArchive

from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.algorithms.moo import nsga2, nsga3
from pymoo.util.ref_dirs import get_reference_directions

from MOO.MOEA import MOEA, NSGA2
from MOO.paretofrontevaluation import ParetoOptimization
from utilities.util import *

dimensions = 100
ga_run = 100
population = 1001
n_objs = [10]
problems = ["WFG7"]
ks = [0.4]
ls = [0.5]

current_working_dir = os.getcwd()
path = re.search(r"^(.*?[\\/]FEA)", current_working_dir)
path = path.group()

for problem in problems:
    if not os.path.isdir(path + "/results/factorarchive/full_solution/" + problem):
        os.mkdir(path + "/results/factorarchive/full_solution/" + problem)
    if not os.path.isdir(path + "/results/factorarchive/full_solution/" + problem + "/NSGA3"):
        os.mkdir(path + "/results/factorarchive/full_solution/" + problem + "/NSGA3")
    if not os.path.isdir(path + "/results/factorarchive/full_solution/" + problem + "/NSGA2"):
        os.mkdir(path + "/results/factorarchive/full_solution/" + problem + "/NSGA2")

    for n_obj in n_objs:
        print(problem, n_obj)
        function = get_problem(problem, n_var=dimensions, n_obj=n_obj)

        reference_point = pickle.load(
            open("E:\\" + problem + "_" + str(n_obj) + "_reference_point.pickle", "rb")
        )
        reference_point = np.array(reference_point)

        po = ParetoOptimization(n_obj)
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=9)
        nsga3_alg = nsga3.NSGA3(ref_dirs=ref_dirs, pop_size=population)

        @add_method(MOEA)
        def calc_fitness(variables, gs=None, factor=None):
            objective_values = function.evaluate(np.array(variables))
            return tuple(objective_values)

        for i in range(3):
            print("#############\n", i, "\n")
            res = minimize(function, nsga3_alg, termination=("n_gen", ga_run))
            # stats_dict = po.evaluate_solution(res.F, reference_point, normalize=True)
            # print('NSGA3: ', stats_dict, len(res.F))

            filename = (
                path
                + "/results/factorarchive/"
                + problem
                + "/NSGA3/NSGA3_9partitions_"
                + problem
                + "_"
                + str(dimensions)
                + "_dim_"
                + str(n_obj)
                + "_objectives_"
                + str(ga_run)
                + "_ea_runs_"
                + str(population)
                + "_population"
                + time.strftime("_%d%m%H%M%S")
                + ".pickle"
            )
            file = open(filename, "wb")
            print(len(res.F))
            pickle.dump(res, file)
            file.close()
        # for k in ks:
        #     for l in ls:
        #         print(k,l)
        #         factorarch = FactorArchive(n_obj, dimensions, percent_best=k, percent_remaining=l)
        #         moo = NSGA2(dimensions=dimensions, value_range=[0.0, 1.0], reference_point=reference_point,
        #                         ea_runs=ga_run, archive=factorarch)
        #         moo.run()
        #         filename = path + '/results/factorarchive/full_solution/'+problem+'/NSGA2/NSGA2_FactorArchive_k_'+str(k).replace('.','')+'_l_'+str(l).replace('.','')+'_'+problem+'_'+str(dimensions)+'_dim_' + str(n_obj) + \
        #                                         '_objectives_'+str(ga_run)+'_ea_runs_'+str(population)+'_population'+time.strftime('_%d%m%H%M%S')+'.pickle'
        #         file = open(filename, "wb")
        #         pickle.dump(moo, file)
        #         file.close()

        # overlapping_archive = moo.nondom_archive.find_archive_overlap(nr_archives_overlapping=2)
        # stats_dict = po.evaluate_solution(overlapping_archive, reference_point, normalize=True)
        # print('Overlapping archive: ', stats_dict)
        # filename = path + '/results/factorarchive/'+problem+'/NSGA2/NSGA2_OverlappingArchive_k_'+str(k).replace('.','')+'_l_'+str(l).replace('.','')+'_'+problem+'_'+str(dimensions)+'_dim_' + str(n_obj) + \
        #                                 '_objectives_'+str(ga_run)+'_ea_runs_'+str(population)+'_population'+time.strftime('_%d%m%H%M%S')+'.pickle'
        # file = open(filename, "wb")
        # pickle.dump(overlapping_archive, file)
        # file.close()

# ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=4)
# nsga3 = nsga3.NSGA3(ref_dirs=ref_dirs, pop_size=population)
# nsga2 = nsga2.NSGA2(pop_size=population)
# res = minimize(function,
#                nsga3,
#                seed=1,
#                termination=('n_gen', ga_run))
# stats_dict = po.evaluate_solution(res.F, reference_point, normalize=True)
# print('NSGA3: ', stats_dict, len(res.F))
#
# filename = path + '/results/factorarchive/'+problem+'/NSGA3/NSGA3_'+problem+'_'+str(dimensions)+'_dim_' + str(n_obj) + \
#                                  '_objectives_'+str(ga_run)+'_ea_runs_'+str(population)+'_population'+time.strftime('_%d%m%H%M%S')+'.pickle'
# file = open(filename, "wb")
# pickle.dump(res.F, file)
# file.close()

# moo = NSGA2(dimensions=dimensions, value_range=[0.0, 1.0], reference_point=list(np.ones(nr_obj)),
#                 ea_runs=ga_run)
# moo.run()

# fa = FactorArchive(nr_obj)
# fa.update_archive(moo.nondom_pop)
