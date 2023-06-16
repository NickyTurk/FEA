from itertools import combinations

from pymoo.core.result import Result

from MOO.MOEA import MOEA
from MOO.archivemanagement import ObjectiveArchive
from utilities.multifilereader import MultiFileReader
import pickle, re, os
from MOO.paretofrontevaluation import ParetoOptimization
from pymoo.indicators.hv import HV
from hvwfg import *
from scipy.stats import kruskal, mannwhitneyu
import numpy as np

"""
Get HV and spread stats for MOO based on however many experimental runs were completed.
"""

problems =  ['henrys']
algorithms = ['SNSGA2', 'CCNSGA2', 'FNSGA2']  # , 'SPEA2', 'MOEAD', 'MOEADPBI']
nr_objs = [3]
parameters = [''] #, 'population_500', 'linear_100_100', 'linear_100_80', 'classic_random_100', 'classic_random_overlap_100_100','diff_grouping_MOO'
file_regex = 'henrys'
directory_to_search = 'D:\\Prescriptions\\RF_optimized\\'
import csv

# field names
# fields = ['problem', 'n_obj', 'algorithm', 'decomp', 'hv', 'hv_std', 'spread', 'sprd_std', 'nd_size', 'n_experiments']
algdecomp = []

# data rows of csv file
for n_obj in nr_objs:
    po = ParetoOptimization(obj_size=n_obj)
    rows = []
    for problem in problems:
        # reference_point = pickle.load(
        #     open('E:\\' + problem + '_' + str(n_obj) + '_reference_point.pickle', 'rb'))
        # reference_point = np.array(reference_point)
        reference_point = None
        t_test_pop = dict()
        # t_test_pop["HV"] = []
        # t_test_pop["spread"] = []
        total_nondom_pop = []

        for alg in algorithms:
            print('************************************************\n', alg, problem, n_obj)
            for param in parameters:
                full_alg_name = alg + '_' + param
                algdecomp.append(full_alg_name)
                # file_regex = alg + r'_(.*)' + problem + r'_(.*)' + re.escape(str(n_obj)) + r'_objectives_(.*)'
                stored_files = MultiFileReader(file_regex=file_regex, dir=directory_to_search)
                file_list = stored_files.path_to_files
                experiments = [file for file in file_list if alg in file]
                # t_test_pop[name] = dict()
                t_test_pop[full_alg_name] = dict()
                t_test_pop[full_alg_name]["HV"] = []
                t_test_pop[full_alg_name]["spread"] = []
                ND_size = 0
                amount = 0
                if len(experiments) != 0:
                    for file in experiments:
                        try:
                            object = pickle.load(open(file, 'rb'))
                        except EOFError:
                            continue
                            #print("error in file: ", file)
                        if alg == 'MOEAD':
                            arch = np.array([np.array(sol.fitness) for sol in object.nondom_archive])
                            if np.any(arch < 0):
                                continue
                        amount += 1
                        if reference_point is not None:
                            if isinstance(object, MOEA):
                                print("MOEA")
                                normalized_archive = np.array(
                                    [np.array(sol.fitness) / reference_point for sol in object.nondom_archive])
                            elif isinstance(object, ObjectiveArchive):
                                normalized_archive = np.array(
                                    [np.array(sol.fitness) / reference_point for sol in object.flatten_archive()])
                            elif isinstance(object, Result):
                                normalized_archive = np.array(
                                    [np.array(sol) / reference_point for sol in object.F])
                            else:
                                normalized_archive = np.array(
                                    [np.array(sol) / reference_point for sol in object])
                            new_div = po.calculate_diversity(normalized_archive,
                                                             normalized=True)  # , minmax=[[0, maxval] for maxval in reference_point])
                            new_hv = wfg(normalized_archive, np.ones(n_obj))
                            ND_size = ND_size + len(normalized_archive)
                        else:
                            new_div = object.iteration_stats[-1]['diversity']
                            new_hv = object.iteration_stats[-1]['hypervolume']
                        t_test_pop[full_alg_name]["spread"].append(new_div)
                        t_test_pop[full_alg_name]["HV"].append(new_hv)
                        ND_size = ND_size + object.iteration_stats[-1]['ND_size']
                    hv_value = np.sum(t_test_pop[full_alg_name]["HV"]) / amount
                    hv_stddev = np.std(t_test_pop[full_alg_name]["HV"])
                    ND_size = ND_size/amount
                    spread = np.sum(t_test_pop[full_alg_name]["spread"]) / amount
                    sprd_stddev = np.std(t_test_pop[full_alg_name]["spread"])
                    # rows.append([problem, str(n_obj), name, decomp, str(hv_value), str(hv_stddev), str(spread), str(sprd_stddev) ,str(ND_size), str(amount)])
                    print('hv: ', hv_value, hv_stddev, 'nd: ', ND_size, 'spread: ', spread, sprd_stddev) #'hv: ', hv_value, hv_stddev, 'nd: ', ND_size, , 'spread: ', spread, sprd_stddev
            print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n')
        # print(kruskal(*t_test_pop["spread"]))
        # print(kruskal(*t_test_pop["HV"]))
        # param_perm = combinations(algdecomp, 2)
        # for perm in param_perm:
        #     print(perm)
        #     print(mannwhitneyu(t_test_pop[perm[0]]["spread"], t_test_pop[perm[1]]["spread"]))
        #     print(mannwhitneyu(t_test_pop[perm[0]]["HV"], t_test_pop[perm[1]]["HV"]))
        #     print('\n\n')

    # name of csv file
    filename = str(n_obj) + "_DTLZ_5_6_normalized_objectives_statistics.csv"

    # writing to csv file
    # if os.path.isfile(filename):
    #     with open(filename, 'a') as csvfile:
    #         csvwriter = csv.writer(csvfile)
    #         #csvwriter.writerow(fields)
    #         csvwriter.writerows(rows)
    #
    # else:
    # with open(filename, 'w', newline='') as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     csvwriter.writerow(fields)
    #     csvwriter.writerows(rows)

"""
TO GET UNION FRONT CALCS
"""

# nondom_fitnesses = []
# for x in total_nondom_pop:
#     try:
#         nondom_fitnesses.append(PopulationMember(x.variables,x.fitness))
#     except:
#         nondom_fitnesses.append(PopulationMember([c.nitrogen for c in x.variables],x.objective_values))
# nondom_indeces = find_non_dominated(np.array([np.array(x.fitness) for x in nondom_fitnesses]))
#     #np.array([np.array(x.fitness) for x in total_nondom_pop]))
# nondom_pop = [nondom_fitnesses[i] for i in nondom_indeces]
# dict_ = po.calculate_diversity(nondom_pop) #po.evaluate_solution(nondom_pop, [1,1,1])
# print("union div: ", dict_)
