from refactoring.utilities.multifilereader import MultiFileReader
import pickle
from refactoring.MOO.paretofront import ParetoOptimization
import numpy as np
from itertools import combinations
from scipy.stats import ttest_ind

file_regexes = [r'_single_knapsack_3_objectives_']
file_regex = r'CCNSGA2_multi_knapsack_3_objectives_'

for file_regex in file_regexes:
    # print(file_regex)
    stored_files = MultiFileReader(file_regex)
    file_list = stored_files.path_to_files
    print(file_list)

    parameters = ['population_500', 'grouping_100_100', 'grouping_200_200', 'grouping_100_80', 'grouping_200_160']
    t_test_pop = dict()
    total_nondom_pop = []

    for param in parameters:
        t_test_pop[param] = dict()
        t_test_pop[param]["HV"] = []
        t_test_pop[param]["spread"] = []
        ND_size = 0
        amount = 0
        for file in file_list:
            if str(param) in file:
                #print(file)
                amount += 1
                obj = pickle.load(open(file, 'rb'))
                t_test_pop[param]["spread"].append(obj.iteration_stats[-1]['diversity'])
                t_test_pop[param]["HV"].append(obj.iteration_stats[-1]['hypervolume'])
                ND_size = ND_size + obj.iteration_stats[-1]['ND_size']
                # total_nondom_pop.extend(obj.nondom_archive)
        HV = (np.sum(t_test_pop[param]["HV"])/amount)/1e14
        ND_size = ND_size/amount
        spread = np.sum(t_test_pop[param]["spread"])/amount
        # #print(file_regex)
        print('------------------------------------------------\n', param)
        print('hv: ', HV, 'nd: ', ND_size, 'spread: ', spread)
    # po = ParetoOptimization()
    # from pymoo.util.nds.non_dominated_sorting import find_non_dominated
    # nondom_indeces = find_non_dominated(
    #     np.array([np.array(x.fitness) for x in total_nondom_pop]))
    # nondom_pop = [total_nondom_pop[i] for i in nondom_indeces]
    # dict_ = po.evaluate_solution(nondom_pop, [1,1,1])
    # print(dict_)

    # param_perm = combinations(parameters, 2)
    # for perm in param_perm:
    #     print(perm)
    #     print(ttest_ind(t_test_pop[perm[0]]["spread"], t_test_pop[perm[1]]["spread"]))
    #     print(ttest_ind(t_test_pop[perm[0]]["HV"], t_test_pop[perm[1]]["HV"]))


