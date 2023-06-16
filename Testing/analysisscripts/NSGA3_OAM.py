import pickle

import numpy as np
from hvwfg import wfg
from matplotlib import pyplot as plt
from pymoo.core.result import Result
from pymoo.util.nds.non_dominated_sorting import find_non_dominated

from MOO.MOEA import MOEA
from MOO.archivemanagement import *
from MOO.paretofrontevaluation import ParetoOptimization
from utilities.multifilereader import MultiFileReader

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

"""
Get plots and stats (HV and spread) for OAM as applied to NSGA3 and NSGA3
"""

problems = ['WFG7'] #  'DTLZ6', 'WFG3', 'WFG7'
ks = [.2] #[.5, .5, .5, .5, .4, .4, .4, .4, .3, .3, .3, .3, .2, .2, .2, .2]
ls = [.5] #[.5, .4, .3, .2, .5, .4, .3, .2, .5, .4, .3, .2, .5, .4, .3, .2]
comparing = []
obj = 10
po = ParetoOptimization(obj_size=obj)
archive_overlap = 6
create_plot = True
calc_stats = False
overall_max = 2.25

for i, problem in enumerate(problems):
    part = 6
    print("******************\n", problem, "\n***********************\n")
    for k, l in zip(ks, ls):
        comparing.append(['NSGA3_'+str(part), 'k_' + str(k).replace('.', '') + '_l_' + str(l).replace('.', '')])
        reference_point = pickle.load(
            open('E:\\reference_points\\' + problem + '_' + str(obj) + '_reference_point.pickle', 'rb'))
        file_regex = r'NSGA3_'+str(part)+'(.*)' + problem + r'_(.*)' + str(obj) + r'_objectives_(.*)'
        stored_files = MultiFileReader(file_regex,
                                       dir="C:\\Users\\amy_l\\PycharmProjects\\FEA\\results\\factorarchive\\")
        experiment_filenames = stored_files.path_to_files
        # for j, compare in enumerate(comparing):
        experiments = [x for x in experiment_filenames if comparing[i][0] in x and problem in x]
        print(experiments)
        print("******************\n")
        print('k: ', k, 'l: ', l)
        avg_hv = {'nsga3':[], 'FAM':[]}
        avg_div = {'nsga3':[], 'FAM':[]}
        lengths = {'nsga3': [], 'FAM': []}
        for id,experiment in enumerate(experiments):
            try:
                results = pickle.load(open(experiment, 'rb'))
            except EOFError:
                print('issues with file: ', experiment)
                continue
            if isinstance(results, Result):
                fa = ObjectiveArchive(obj, dimensions=100, percent_best=k, percent_diversity=l)
                fa.update_archive(results)
                lengths['nsga3'].append(len(results.F))
                overlapping_archive = fa.find_archive_overlap(nr_archives_overlapping=archive_overlap)
                lengths['FAM'].append(len(overlapping_archive))
            else:
                continue
            if len(overlapping_archive) > 2 and calc_stats:
                # sol_set = environmental_solution_selection_nsga2(results.flatten_archive(),
                #                                                      sol_size=len(overlapping_archive))
                overlapping_fitness = np.array([np.array(sol.fitness) / reference_point for sol in overlapping_archive])
                nsga3_fitness = np.array([np.array(sol) / reference_point for sol in results.F])
                overlapping_hv = wfg(overlapping_fitness, np.ones(obj))
                nsga3_hv = wfg(nsga3_fitness, np.ones(obj))
                overlapping_div = po.calculate_diversity(overlapping_fitness,
                                                         normalized=True)
                nsga3_div = po.calculate_diversity(nsga3_fitness, normalized=True)
                avg_hv['nsga3'].append(nsga3_hv)
                avg_hv['FAM'].append(overlapping_hv)
                avg_div['nsga3'].append(nsga3_div)
                avg_div['FAM'].append(overlapping_div)

            if len(overlapping_archive) > 0 and create_plot:
                try:
                    max_value = np.max(np.array([x.fitness / reference_point for x in overlapping_archive]))
                except AttributeError:
                    max_value = np.max(np.array([x / reference_point for x in overlapping_archive]))
                # if max_value > overall_max:
                #     overall_max = max_value
                fig = plt.figure()
                ax = fig.add_subplot(projection='polar')
                for sol in overlapping_archive:
                    try:
                        solution = np.array([x for x in sol.fitness / reference_point])
                    except AttributeError:
                        solution = np.array([x for x in sol / reference_point])
                    obj_dict = dict()
                    for i, fitness in enumerate(solution):
                        keystring = "Objective " + str(i + 1)
                        obj_dict[keystring] = [fitness]
                    df = pd.DataFrame(obj_dict)

                    # calculate values at different angles
                    z = df.rename(index={0: 'value'}).T.reset_index()
                    z = z.append(z.iloc[0], ignore_index=True)  #
                    z = z.reindex(np.arange(z.index.min(), z.index.max() + 1e-10, 0.05))

                    z['angle'] = np.linspace(0, 2 * np.pi, len(z))
                    z.plot.scatter('angle', 'value', ax=ax, legend=False)
                    z['value'] = z['value'].interpolate(method='linear')  # method='linear'

                    # plot
                    # fig = plt.figure()
                    # ax = fig.add_subplot(projection='polar')
                    z.plot('angle', 'value', ax=ax, legend=False)
                    ax.fill_between(z['angle'], 0, z['value'], alpha=0.1)
                    ax.set_xticks(z.dropna()['angle'].iloc[:-1])
                    ax.set_xticklabels(z.dropna()['index'].iloc[:-1])
                    ax.set_xlabel(None)
                    ax.set_ylabel(None)
                    tickvalues = [round(j * overall_max / 5, 3) for j in range(5)]
                    ax.set_yticks(tickvalues)
                    ax.set_rlabel_position(-65)

                # striped background
                n = 5
                vmax = overall_max
                for i in np.arange(n):
                    ax.fill_between(
                        np.linspace(0, 2 * np.pi, 100), vmax / n / 2 * i * 2, vmax / n / 2 * (i * 2 + 1),
                        color='silver', alpha=0.1)
                plt.title(problem)
                plt.show()
                pathtosave = "./objective_polarplots/" + problem + "/NGSA3/"
                fig.savefig(pathtosave+problem + "_" + str(obj) + "-obj_NSGA3_OAM-S_k"+str(k).replace('.','')+"_l"+str(l).replace('.','')+"_"+str(id))
        print("nsga3: ", np.mean(avg_hv['nsga3']), np.std(avg_hv['nsga3']), np.mean(avg_div['nsga3']),
              np.std(avg_div['nsga3']))
        print("nsga3 FAM: ", np.mean(avg_hv['FAM']), np.std(avg_hv['FAM']), np.mean(avg_div['FAM']), np.std(avg_div['FAM']))
        print("og vs new lengths: ", np.mean(lengths['nsga3']), np.mean(lengths['FAM']))