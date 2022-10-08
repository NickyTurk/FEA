from utilities.multifilereader import MultiFileReader
import pickle, re
from MOO.paretofrontevaluation import ParetoOptimization
from utilities.util import PopulationMember
import numpy as np
from itertools import combinations
from scipy.stats import ttest_ind
from pymoo.util.nds.non_dominated_sorting import find_non_dominated

#TODO: make reusable and general

file_regexes = [r'', r'']
file_regex = r'CCNSGA2_multi_knapsack_3_objectives_'

problems = ['DTLZ2', 'DTLZ3']
algorithms = ['NSGA2', 'SPEA2', 'MOEAD']
nr_objs = [3,5, 10]

po = ParetoOptimization()

for problem in problems:
    print('************************************************')
    print(problem)
    for obj in nr_objs:
        print(obj)
        file_regex = re.escape(problem)+r'_1000_dimensions_'+ re.escape(str(obj))+r'_objectives_ea_runs_(.*)_population_500_'
        #file_regex = r'_1000_dimensions_3_objectives_ea_runs_[0-9]*_population_500_'

        t_test_pop = dict()
        total_nondom_pop = []
        
        for alg in algorithms:
            print('************************************************')
            stored_files = MultiFileReader(file_regex=file_regex, dir = "/media/amy/WD Drive/"+problem+"/"+alg+"/")
            file_list = stored_files.path_to_files
            t_test_pop[alg] = dict()
            t_test_pop[alg]["HV"] = []
            t_test_pop[alg]["spread"] = []
            ND_size = 0
            amount = 0
            for file in file_list:
                print(file)
                amount += 1
                obj = pickle.load(open(file, 'rb'))
                new_div = po.calculate_diversity(obj.nondom_archive)
                t_test_pop[alg]["spread"].append(new_div) #obj.iteration_stats[-1]['diversity'])
                t_test_pop[alg]["HV"].append(obj.iteration_stats[-1]['hypervolume'])
                ND_size = ND_size + obj.iteration_stats[-1]['ND_size']
                total_nondom_pop.extend(obj.nondom_archive)
            HV = np.sum(t_test_pop[alg]["HV"])/amount
            ND_size = ND_size/amount
            spread = np.sum(t_test_pop[alg]["spread"])/amount
            # #print(file_regex)
            print('\n------------------------------------------------\n', alg)
            print('hv: ', HV, 'nd: ', ND_size, 'spread: ', spread)
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
        print('**************************************************\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\n')

        param_perm = combinations(algorithms, 2)
        for perm in param_perm:
            print(perm)
            print(ttest_ind(t_test_pop[perm[0]]["spread"], t_test_pop[perm[1]]["spread"]))
            print(ttest_ind(t_test_pop[perm[0]]["HV"], t_test_pop[perm[1]]["HV"]))


