from refactoring.utilities.multifilereader import MultiFileReader
import pickle, random
import numpy as np
from pymoo.util.nds.non_dominated_sorting import find_non_dominated
from itertools import combinations

#TODO: make reusable and general

nondom_solutions = dict()
lengths = dict()
comparing = ['population_500', 'grouping_100_100', 'grouping_100_80', 'grouping_200_200', 'grouping_200_160']
total_front = []

file_regex = r'_single_knapsack_3_objectives_'
stored_files = MultiFileReader(file_regex)
experiment_filenames = stored_files.path_to_files

for compare in comparing:
    experiment = [x for x in experiment_filenames if compare+'_' in x]
    if experiment:
        rand_int = random.randint(0, len(experiment)-1)
        print(rand_int)
        experiment = experiment[rand_int]
    else:
        break
    feamoo = pickle.load(open(experiment, 'rb'))
    solutions = np.array([np.array(x.fitness) for x in feamoo.nondom_archive])
    total_front.extend(solutions)
    nondom_solutions[compare] = solutions
    lengths[compare] = len(solutions)
    print(compare, ' len of front', len(solutions))

print('total front length ', len(total_front))
#total_front = np.vstack((nondom_solutions['CCEA'], nondom_solutions['FEA'], nondom_solutions['NSGA']))
indeces = find_non_dominated(np.array(total_front))

lb = 0
ub = 0
print('Method VS TOTAL front:')
for i, compare in enumerate(comparing):
    print(compare)
    ub += lengths[compare]
    if i != 0:
        lb += lengths[comparing[i-1]]
    #print('upper and lower: ', ub, lb)
    print(len([x for x in indeces if lb <= x < ub]) / len(indeces))

pair_compare = [comb for comb in combinations(comparing, 2)]
for pair in pair_compare:
    print(pair)
    to_compare = np.vstack((nondom_solutions[pair[0]], nondom_solutions[pair[1]]))
    indeces = find_non_dominated(to_compare)
    print(pair[0], len([x for x in indeces if x < lengths[pair[0]]])/lengths[pair[0]])
    print(pair[1], len([x for x in indeces if lengths[pair[0]] <= x <= lengths[pair[0]]+lengths[pair[1]]])/lengths[pair[1]])