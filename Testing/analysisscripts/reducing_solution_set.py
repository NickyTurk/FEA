import pickle

from hvwfg import wfg
from pymoo.util.nds.non_dominated_sorting import find_non_dominated

from MOO.archivemanagement import *
from utilities.multifilereader import MultiFileReader

problems = ['DTLZ5', 'DTLZ6', 'WFG3', 'WFG7']  # ['DTLZ1', 'DTLZ2', 'DTLZ3', 'DTLZ4', 'DTLZ5', 'DTLZ6'] #, 'WFG1', 'WFG2', 'WFG3', 'WFG4', 'WFG5', 'WFG7']
comparing = [['NSGA2', 'k_05_l_02'], ['NSGA2', 'k_05_l_03'],['NSGA2', 'k_05_l_04'],['NSGA2', 'k_05_l_05'],
             ['NSGA2', 'k_04_l_02'], ['NSGA2', 'k_04_l_03'],['NSGA2', 'k_04_l_04'],['NSGA2', 'k_04_l_05'],
             ['NSGA2', 'k_025_l_02'], ['NSGA2', 'k_025_l_03'],['NSGA2', 'k_025_l_04'],['NSGA2', 'k_025_l_05']]
obj = 5

for i,problem in enumerate(problems):
    reference_point = pickle.load(
        open('D:\\' + problem + '\\' + problem + '_' + str(obj) + '_reference_point.pickle', 'rb'))
    file_regex = r'_' + problem + r'_(.*)' + str(obj) + r'_objectives_'
    stored_files = MultiFileReader(file_regex,
                                   dir="C:\\Users\\amy_l\\PycharmProjects\\FEA\\results\\factorarchive\\full_solution\\" + problem + "\\")
    experiment_filenames = stored_files.path_to_files
    for j,compare in enumerate(comparing):
        experiments = [x for x in experiment_filenames if compare[0] in x and compare[1] in x]
        for experiment in experiments:
            try:
                results = pickle.load(open(experiment, 'rb'))
            except EOFError:
                print('issues with file: ', experiment)
                continue
            # archive = np.array([np.array(x.fitness) for x in results.nondom_pop])
            overlapping_archive = results.nondom_archive.find_archive_overlap(nr_archives_overlapping=4)
            print("archive length: ", len(overlapping_archive))
            if len(overlapping_archive) > 2:
                sol_set = environmental_solution_selection_nsga2(results.nondom_pop, sol_size=len(overlapping_archive))
                environmental_fitness = np.array([np.array(sol.fitness)/reference_point for sol in sol_set])
                overlapping_fitness = np.array([np.array(sol.fitness)/reference_point for sol in overlapping_archive])
                environmental_hv = wfg(environmental_fitness, np.ones(obj))
                overlapping_hv = wfg(overlapping_fitness, np.ones(obj))
                print(environmental_hv, overlapping_hv)

                total_front = []
                total_front.extend(environmental_fitness)
                total_front.extend(overlapping_fitness)
                indeces = find_non_dominated(np.array(total_front))
                lb = 0
                ub = len(environmental_fitness)
                print(len([x for x in indeces if lb <= x < ub]) / len(indeces))
                lb = len(environmental_fitness)
                ub += len(overlapping_fitness)
                print(len([x for x in indeces if lb <= x < ub]) / len(indeces))