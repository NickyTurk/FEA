from itertools import combinations

from utilities.multifilereader import MultiFileReader
import pickle, re, os
from MOO.paretofrontevaluation import ParetoOptimization
from pymoo.indicators.hv import HV
from hvwfg import *
from scipy.stats import kruskal, mannwhitneyu
import numpy as np

problems =  ['WFG7']
algorithms = ['NSGA2', 'SPEA2', 'MOEAD']  # , 'SPEA2', 'MOEAD', 'MOEADPBI']
nr_objs = [5, 10]
decompositions = ['population_500', 'linear_100_100', 'linear_100_80', 'classic_random_100', 'classic_random_overlap_100_100', 'diff_grouping_MOO'] #, 'population_500', 'linear_100_100', 'linear_100_80', 'classic_random_100', 'classic_random_overlap_100_100','diff_grouping_MOO'

import csv

# field names
# fields = ['problem', 'n_obj', 'algorithm', 'decomp', 'hv', 'hv_std', 'spread', 'sprd_std', 'nd_size', 'n_experiments']
algdecomp = []

# data rows of csv file
for n_obj in nr_objs:
    po = ParetoOptimization(obj_size=n_obj)
    rows = []
    for problem in problems:
        reference_point = pickle.load(
            open('D:\\' + problem + '\\' + problem + '_' + str(n_obj) + '_reference_point.pickle', 'rb'))
        reference_point = np.array(reference_point)
        t_test_pop = dict()
        # t_test_pop["HV"] = []
        t_test_pop["spread"] = []
        total_nondom_pop = []

        for alg in algorithms:
            print('************************************************\n', alg, problem, n_obj)
            for decomp in decompositions:
                if decomp == "linear_100_100" or decomp == "classic_random_100":
                    name = "CC" + alg
                elif decomp == "population_500":
                    name = alg
                else:
                    name = "F" + alg
                algdecompname = name + '_' + decomp
                algdecomp.append(algdecompname)
                file_regex = name + r'_' + problem + r'_(.*)' + re.escape(str(n_obj)) + r'_objectives_(.*)' + decomp
                stored_files = MultiFileReader(file_regex=file_regex, dir="D:/" + problem + "/" + name + "/")
                file_list = stored_files.path_to_files
                # t_test_pop[name] = dict()
                t_test_pop[algdecompname] = dict()
                t_test_pop[algdecompname]["HV"] = []
                t_test_pop[algdecompname]["spread"] = []
                ND_size = 0
                amount = 0
                if len(file_list) != 0:
                    for file in file_list:
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
                        normalized_archive = np.array(
                            [np.array(sol.fitness) / reference_point for sol in object.nondom_archive])
                        new_div = po.calculate_diversity(normalized_archive,
                                                         normalized=True)  # , minmax=[[0, maxval] for maxval in reference_point])
                        hv = HV(ref_point=np.ones(n_obj))
                        #new_hv = hv(normalized_archive)
                        new_hv = wfg(normalized_archive, np.ones(n_obj))
                        #print('hv: ', new_hv)
                        t_test_pop[algdecompname]["spread"].append(new_div)
                        t_test_pop[algdecompname]["HV"].append(new_hv)
                        ND_size = ND_size + object.iteration_stats[-1]['ND_size']
                        # total_nondom_pop.extend(obj.nondom_archive)
                    #t_test_pop["HV"].append(t_test_pop[algdecompname]["HV"])
                    #t_test_pop["spread"].append(t_test_pop[algdecompname]["spread"])
                    hv_value = np.sum(t_test_pop[algdecompname]["HV"])/amount
                    hv_stddev = np.std(t_test_pop[algdecompname]["HV"])
                    ND_size = ND_size/amount
                    spread = np.sum(t_test_pop[algdecompname]["spread"])/amount
                    sprd_stddev = np.std(t_test_pop[algdecompname]["spread"])
                    # rows.append([problem, str(n_obj), name, decomp, str(hv_value), str(hv_stddev), str(spread), str(sprd_stddev) ,str(ND_size), str(amount)])
                    print('hv: ', hv_value, hv_stddev, 'nd: ', ND_size) #'hv: ', hv_value, hv_stddev, 'nd: ', ND_size, , 'spread: ', spread, sprd_stddev
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
