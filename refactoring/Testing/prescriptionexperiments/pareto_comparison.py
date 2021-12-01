import pickle
import numpy as np
import itertools
#from pygmo.core import hypervolume
from pymoo.util.nds.non_dominated_sorting import find_non_dominated
from refactoring.MOO.paretofront import ParetoOptimization

experiment_filenames = [
    "../../results/FEAMOO/CCEAMOO_Henrys_trial_3_objectives_linear_topo_ga_runs_100_population_500_0508121527.pickle",
    "../../results/FEAMOO/CCEAMOO_Sec35Middle_trial_3_objectives_strip_topo_ga_runs_100_population_500_3007121518.pickle",
    "../../results/FEAMOO/CCEAMOO_Sec35West_trial_3_objectives_strip_topo_ga_runs_100_population_500_0408143024.pickle",
    "../../results/FEAMOO/NSGA2_Henrys_trial_3_objectives_ga_runs_200_population_500_2807110247.pickle",
    "../../results/FEAMOO/NSGA2_Sec35Middle_trial_3_objectives_ga_runs_200_population_500_2807110338.pickle",
    "../../results/FEAMOO/NSGA2_Sec35West_trial_3_objectives_ga_runs_200_population_500_2807110402.pickle",
    "../../results/FEAMOO/FEAMOO_Sec35West_trial_3_objectives_strip_topo_ga_runs_100_population_500_0808133844.pickle",
    "../../results/FEAMOO/FEAMOO_Sec35Middle_trial_3_objectives_linear_topo_ga_runs_100_population_500_2807191458.pickle",
    "../../results/FEAMOO/FEAMOO_Henrys_trial_3_objectives_strip_topo_ga_runs_100_population_500_1008025822.pickle"]

field_names = [ 'sec35west','sec35middle', 'henrys']
methods = ["CCEAMOO", "NSGA2", "FEAMOO"]
nondom_solutions = dict()
for field_name in field_names:
    full_solutions = []
    print(field_name)
    nondom_solutions[field_name] = dict()
    for method in methods:
        print(method)
        experiment = [x for x in experiment_filenames if method+'_' in x and field_name in x.lower()]
        if experiment:
            experiment = experiment[0]
        else:
            break
        feamoo = pickle.load(open(experiment, 'rb'))
        print(feamoo.iteration_stats[-1])
        nondom_solutions[field_name][method] = np.array([np.array(x.objective_values) for x in feamoo.nondom_archive])
        full_solutions.extend([x for x in feamoo.nondom_archive])
    ccea_len = len(nondom_solutions[field_name]['CCEAMOO'])
    fea_len = len(nondom_solutions[field_name]['FEAMOO'])
    nsga_len = len(nondom_solutions[field_name]['NSGA2'])

    total_front = np.vstack((nondom_solutions[field_name]['CCEAMOO'], nondom_solutions[field_name]['FEAMOO'], nondom_solutions[field_name]['NSGA2']))
    indeces = find_non_dominated(total_front)
    global_solutions = [total_front[i] for i in indeces]
    all_solutions = [full_solutions[i] for i in indeces]
    #hv = hypervolume(np.array(global_solutions))
    po = ParetoOptimization()
    print('diversity: ', po.calculate_diversity(all_solutions))
    #print('hypervolume: ', hv.compute(ref_point=np.array([1,1,1])))
    print('CCEA value', len([x for x in indeces if x < ccea_len]) / len(indeces))
    print('FEA value', len([x for x in indeces if  ccea_len <= x < ccea_len+fea_len])/len(indeces))
    print('NSGA value', len([x for x in indeces if ccea_len+fea_len <= x <= ccea_len + fea_len + nsga_len]) / len(indeces))

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