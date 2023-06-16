import pickle
import numpy as np
import itertools, random
#from pygmo.core import hypervolume
from pymoo.util.nds.non_dominated_sorting import find_non_dominated
from MOO.paretofrontevaluation import ParetoOptimization
from utilities.multifilereader import MultiFileReader

# TODO: generalize

mf = MultiFileReader(dir="D:\\Prescriptions\\CNN_optimized\\", file_regex="henrys")
experiment_filenames = mf.path_to_files

methods = ["SNSGA2", "CCNSGA2", "FNSGA2"]
fea_sum = 0
nsga_sum = 0
for i in range(5):
    nondom_solutions = dict()
    full_solutions = []
    for method in methods:
        experiment = [x for x in experiment_filenames if method in x.upper()]
        if experiment:
            experiment = experiment[random.randint(0,len(experiment)-1)]
        else:
            break
        feamoo = pickle.load(open(experiment, 'rb'))
        if feamoo.nondom_archive:
            nondom_solutions[method] = np.array([np.array(x.fitness) for x in feamoo.nondom_archive])
            full_solutions.extend([x for x in feamoo.nondom_archive])
        else:
            nondom_solutions[method] = np.array([np.array(x.fitness) for x in feamoo.nondom_pop])
            full_solutions.extend([x for x in feamoo.nondom_pop])
    ccea_len = len(nondom_solutions['CCNSGA2'])
    nsga_len = len(nondom_solutions['SNSGA2'])
    fea_len = len(nondom_solutions['FNSGA2'])

    total_front = np.vstack((nondom_solutions['CCNSGA2'], nondom_solutions['FNSGA2'], nondom_solutions['SNSGA2']))
    indeces = find_non_dominated(total_front)
    global_solutions = [total_front[i] for i in indeces]
    all_solutions = [full_solutions[i] for i in indeces]
    po = ParetoOptimization()
    print('hv and diversity: ', po.evaluate_solution(all_solutions, [1,1,1]))
    print('CCEA value', len([x for x in indeces if x < ccea_len]) / len(indeces))
    fea_coverage = len([x for x in indeces if  ccea_len <= x < ccea_len+fea_len])/len(indeces)
    fea_sum += fea_coverage
    print('FEA value', fea_coverage)
    nsga_coverage = len([x for x in indeces if ccea_len+fea_len <= x <= ccea_len + fea_len + nsga_len]) / len(indeces)
    nsga_sum += nsga_coverage
    print('NSGA value', nsga_coverage)

print("averages:", fea_sum/5, nsga_sum/5 )

# print('CCEAMOO vs FEAMOO for field ', field_name)
# to_compare = np.vstack((nondom_solutions[field_name]['CCEAMOO'], nondom_solutions[field_name]['FEAMOO']))
# indeces = find_non_dominated(to_compare)
# print('CCEA value', len([x for x in indeces if x < ccea_len])/ccea_len)
# print('FEA value', len([x for x in indeces if  ccea_len <= x <= ccea_len+fea_len])/fea_len)
#
# print('CCEAMOO vs NSGA2 for field ', field_name)
# to_compare = np.vstack((nondom_solutions[field_name]['CCEAMOO'], nondom_solutions[field_name]['NSGA2']))
# indeces = find_non_dominated(to_compare)
# print('CCEA value', len([x for x in indeces if x < ccea_len])/ccea_len)
# print('NSGA value', len([x for x in indeces if  ccea_len <= x <= ccea_len+nsga_len])/nsga_len)
#
# print('NSGA2 vs FEAMOO for field ', field_name)
# to_compare = np.vstack((nondom_solutions[field_name]['NSGA2'], nondom_solutions[field_name]['FEAMOO']))
# indeces = find_non_dominated(to_compare)
# print('NSGA value', len([x for x in indeces if x < nsga_len])/nsga_len)
# print('FEA value', len([x for x in indeces if  nsga_len <= x <= nsga_len+fea_len])/fea_len)