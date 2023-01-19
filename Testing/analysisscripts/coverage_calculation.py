from utilities.multifilereader import MultiFileReader
import pickle, random
import numpy as np
from pymoo.util.nds.non_dominated_sorting import find_non_dominated
from itertools import combinations

algorithms = ['NSGA2', 'NSGA3']  # MOEAD, SPEA2
objs = [5]
problems = ['DTLZ6']  # ['DTLZ1', 'DTLZ2', 'DTLZ3', 'DTLZ4', 'DTLZ5', 'DTLZ6'] #, 'WFG1', 'WFG2', 'WFG3', 'WFG4', 'WFG5', 'WFG7']
comparing = [ ['NSGA2', 'FactorArchive_k_03_l_02'], ['NSGA3', '1000_population']]
archive_overlap = 1

for problem in problems:
    print(problem)
    for obj in objs:
        alg_list = []
        file_regex = r'_' + problem + r'_(.*)' + str(obj) + r'_objectives_'
        stored_files = MultiFileReader(file_regex, dir= "C:\\Users\\amy_l\\PycharmProjects\\FEA\\results\\factorarchive\\" + problem + "\\")
        experiment_filenames = stored_files.path_to_files
        total_front = []
        lengths = dict()
        for compare in comparing:
            full_compare = compare[0] + '_' + compare[1]
            alg_list.append(full_compare)
            experiments = [x for x in experiment_filenames if compare[0] in x and compare[1] in x]  # and 'PBI' not in x
            if experiments:
                rand_int = random.randint(0, len(experiments) - 1)
                experiment = experiments[rand_int]
            else:
                break
            try:
                results = pickle.load(open(experiment, 'rb'))
            except EOFError:
                print('issues with file: ', experiment)
                continue
            try:
                if 'FactorArchive' in compare[1]:
                    results = results.find_archive_overlap(nr_archives_overlapping=archive_overlap)
                total_front.extend(np.array([np.array(x.fitness) for x in results]))
            except AttributeError:
                total_front.extend(np.array([np.array(x) for x in results]))
            lengths[full_compare] = len(results)
            print(len(results))

            # print(compare, ' len of front', len(solutions))

        #print('total front length ', len(total_front))
        # total_front = np.vstack((nondom_solutions['CCEA'], nondom_solutions['FEA'], nondom_solutions['NSGA']))
        indeces = find_non_dominated(np.array(total_front))
        lb = 0
        ub = 0
        for i, compare in enumerate(alg_list):
            try:
                ub += lengths[compare]
            except KeyError:
                print("no results for: ", compare)
            if i != 0:
                try:
                    lb += lengths[alg_list[i - 1]]
                except KeyError:
                    print("no results")
            #print('upper and lower: ', ub, lb)
            print(len([x for x in indeces if lb <= x < ub]) / len(indeces))

            # pair_compare = [comb for comb in combinations(comparing, 2)]
            # for pair in pair_compare:
            #     print(pair)
            #     to_compare = np.vstack((nondom_solutions[pair[0]], nondom_solutions[pair[1]]))
            #     indeces = find_non_dominated(to_compare)
            #     print(pair[0], lengths[pair[0]], len([x for x in indeces if x < lengths[pair[0]]])/lengths[pair[0]])
            #     print(pair[1], lengths[pair[1]], len([x for x in indeces if lengths[pair[0]] <= x <= lengths[pair[0]]+lengths[pair[1]]])/lengths[pair[1]])

        # print('\n------------------------------------------------\n')