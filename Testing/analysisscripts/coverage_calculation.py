from pymoo.core.result import Result

from MOO.MOEA import MOEA
from MOO.archivemanagement import ObjectiveArchive
from utilities.multifilereader import MultiFileReader
import pickle, random
import numpy as np
from pymoo.util.nds.non_dominated_sorting import find_non_dominated
from itertools import combinations

algorithms = ['NSGA2', 'NSGA3']  # MOEAD, SPEA2
objs = [10]
problems = ['WFG7']  # ['DTLZ1', 'DTLZ2', 'DTLZ3', 'DTLZ4', 'DTLZ5', 'DTLZ6'] #, 'WFG1', 'WFG2', 'WFG3', 'WFG4', 'WFG5', 'WFG7']
comparing = [['NSGA2', 'FactorArchive_k_04_l_05'],  ['NSGA3', '4partitions']]
overlap = .6
# comparing = [['NSGA2', 'FactorArchive_k_05_l_02'], ['NSGA2', 'FactorArchive_k_05_l_03'], ['NSGA2', 'FactorArchive_k_05_l_04'],['NSGA2', 'FactorArchive_k_05_l_05'],
#              ['NSGA2', 'FactorArchive_k_04_l_02'], ['NSGA2', 'FactorArchive_k_04_l_03'], ['NSGA2', 'FactorArchive_k_04_l_04'],['NSGA2', 'FactorArchive_k_04_l_05'],
#              ['NSGA2', 'FactorArchive_k_025_l_02'], ['NSGA2', 'FactorArchive_k_025_l_03'], ['NSGA2', 'FactorArchive_k_025_l_04'],['NSGA2', 'FactorArchive_k_025_l_05']]

for problem in problems:
    print(problem)
    for obj in objs:
        archive_overlap = overlap * obj
        print(archive_overlap)
        print(obj)
        average_AC = dict()
        average_len = dict()
        alg_list = []
        for compare in comparing:
            full_compare = compare[0] + '_' + compare[1]
            alg_list.append(full_compare)
            average_AC[full_compare] = []
            average_len[full_compare] = []
        file_regex = r'_' + problem + r'_(.*)' + str(obj) + r'_objectives_'
        stored_files = MultiFileReader(file_regex, dir= "C:\\Users\\amy_l\\PycharmProjects\\FEA\\results\\factorarchive\\" + problem + "\\")
        experiment_filenames = stored_files.path_to_files
        for kfold in range(20):
            total_front = []
            lengths = dict()
            for compare in comparing:
                full_compare = compare[0] + '_' + compare[1]
                experiments = [x for x in experiment_filenames if compare[0] in x and compare[1] in x]  # and 'PBI' not in x
                if experiments:
                    rand_int = random.randint(0, len(experiments) - 1)
                    experiment = experiments[rand_int]
                    try:
                        results = pickle.load(open(experiment, 'rb'))
                    except EOFError:
                        print('issues with file: ', experiment)
                        continue
                    if isinstance(results, ObjectiveArchive):
                        archive = results.find_archive_overlap(nr_archives_overlapping=archive_overlap)
                        total_front.extend(np.array([np.array(x.fitness) for x in archive]))
                    elif isinstance(results, MOEA):
                        try:
                            archive = results.nondom_archive.find_archive_overlap(nr_archives_overlapping=archive_overlap)
                        except AttributeError:
                            archive = results.nondom_archive
                        total_front.extend(np.array([np.array(x.fitness) for x in archive]))
                    elif isinstance(results, Result):
                        archive = results.F
                        total_front.extend(np.array([np.array(x) for x in archive]))
                    else:
                        archive = results
                        total_front.extend(np.array([np.array(x) for x in results]))
                    lengths[full_compare] = len(archive)
                    average_len[full_compare].append(len(archive))
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
                #print('params: ', compare, ' with upper and lower: ', ub, lb, "length = ", lengths[compare])
                AC = len([x for x in indeces if lb <= x < ub]) / len(indeces)
                average_AC[compare].append(AC)
        for name, AC in average_AC.items():
            print(name, np.mean(AC), np.mean(average_len[name]))
            # pair_compare = [comb for comb in combinations(comparing, 2)]
            # for pair in pair_compare:
            #     print(pair)
            #     to_compare = np.vstack((nondom_solutions[pair[0]], nondom_solutions[pair[1]]))
            #     indeces = find_non_dominated(to_compare)
            #     print(pair[0], lengths[pair[0]], len([x for x in indeces if x < lengths[pair[0]]])/lengths[pair[0]])
            #     print(pair[1], lengths[pair[1]], len([x for x in indeces if lengths[pair[0]] <= x <= lengths[pair[0]]+lengths[pair[1]]])/lengths[pair[1]])

# file_regex = r'NSGA3_3partitions_' + problem + r'_(.*)' + str(obj) + r'_objectives_'
# nsga3_files = MultiFileReader(file_regex,
#                                dir="C:\\Users\\amy_l\\PycharmProjects\\FEA\\results\\factorarchive\\")
# nsga3_filenames = nsga3_files.path_to_files
# randnumb = random.randint(0, len(nsga3_filenames)-1)
# nsga3 = pickle.load(open(nsga3_filenames[randnumb], 'rb'))
# try:
#     nsga3_norm = np.array([np.array(x)/reference_point for x in nsga3])
#     nsga3_len = len(nsga3)
# except TypeError:
#     nsga3_norm = np.array([np.array(x) / reference_point for x in nsga3.F])
#     nsga3_len = len(nsga3.F)
# nsga3_div = po.calculate_diversity(nsga3_norm, normalized=True)
# nsga3_hv = wfg(nsga3_norm, np.ones(obj))
# print('\nNSGA3:\nHV:', nsga3_hv, '\nspread: ', nsga3_div, '\nsize: ', nsga3_len)
# print('ES\nHV:', environmental_hv, '\nspread: ', environmental_div, '\nsize: ',len(overlapping_archive))
# print('FAM\nHV:', overlapping_hv, '\nspread: ', overlapping_hv, '\nsize: ',len(overlapping_archive))
#
# total_front = []
# total_front.extend(nsga3_norm)
# total_front.extend(overlapping_fitness)
# indeces = find_non_dominated(np.array(total_front))
# lb = 0
# ub = nsga3_len
# nsga3_AC = len([x for x in indeces if lb <= x < ub]) / len(indeces)
# lb = nsga3_len
# ub += len(overlapping_fitness)
# overlap_AC = len([x for x in indeces if lb <= x < ub]) / len(indeces)
# print("coverage (NS3 vs FAM): ", nsga3_AC, overlap_AC)
# if nsga3_AC == 1:
#     print(nsga3_filenames[randnumb])
#     print(experiment)