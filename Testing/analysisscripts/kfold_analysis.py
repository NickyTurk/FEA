from utilities.multifilereader import MultiFileReader
import pickle, re
from MOO.paretofrontevaluation import ParetoOptimization
from utilities.util import PopulationMember
import numpy as np
from itertools import combinations
from scipy.stats import ttest_ind

#TODO: make reusable and general

file_regexes = [r'_multi_knapsack_5', r'_single_knapsack_5']
file_regex = r'CCNSGA2_multi_knapsack_3_objectives_'
po = ParetoOptimization()

for file_regex in file_regexes:
    print(file_regex)
    stored_files = MultiFileReader(file_regex, dir = "/media/amy/WD Drive/Knapsack/final/")
    file_list = stored_files.path_to_files

    parameters = [r'/nsga2/', r'ccnsga2.*100', r'fnsga2.*100', r'ccnsga2.*200', r'fnsga2.*200']
    t_test_pop = dict()
    total_nondom_pop = []

    for param in parameters:
        t_test_pop[param] = dict()
        t_test_pop[param]["HV"] = []
        t_test_pop[param]["spread"] = []
        ND_size = 0
        amount = 0
        for file in file_list:
            regexp = re.compile(param)
            if regexp.search(file.lower()):#str(param) in file.lower():
                amount += 1
                obj = pickle.load(open(file, 'rb'))
                new_div = po.calculate_diversity(obj.nondom_archive)
                t_test_pop[param]["spread"].append(new_div) #obj.iteration_stats[-1]['diversity'])
                t_test_pop[param]["HV"].append(obj.iteration_stats[-1]['hypervolume'])
                ND_size = ND_size + obj.iteration_stats[-1]['ND_size']
                total_nondom_pop.extend(obj.nondom_archive)
        HV = np.sum(t_test_pop[param]["HV"])/amount
        ND_size = ND_size/amount
        spread = np.sum(t_test_pop[param]["spread"])/amount
        # #print(file_regex)
        print('\n------------------------------------------------\n', param)
        print('hv: ', HV, 'nd: ', ND_size, 'spread: ', spread)
    from pymoo.util.nds.non_dominated_sorting import find_non_dominated
    nondom_fitnesses = []
    for x in total_nondom_pop:
        try: 
            nondom_fitnesses.append(PopulationMember(x.variables,x.fitness))
        except:
            nondom_fitnesses.append(PopulationMember([c.nitrogen for c in x.variables],x.objective_values))
    nondom_indeces = find_non_dominated(np.array([np.array(x.fitness) for x in nondom_fitnesses]))
        #np.array([np.array(x.fitness) for x in total_nondom_pop]))
    nondom_pop = [nondom_fitnesses[i] for i in nondom_indeces]
    dict_ = po.calculate_diversity(nondom_pop) #po.evaluate_solution(nondom_pop, [1,1,1])
    print("union div: ", dict_)

    param_perm = combinations(parameters, 2)
    for perm in param_perm:
        print(perm)
        print(ttest_ind(t_test_pop[perm[0]]["spread"], t_test_pop[perm[1]]["spread"]))
        print(ttest_ind(t_test_pop[perm[0]]["HV"], t_test_pop[perm[1]]["HV"]))


