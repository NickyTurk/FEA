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

problems = ['WFG7'] #  'DTLZ6', 'WFG3', 'WFG7'
ks = [.5]
ls = [.4]
comparing = []
obj = 10
po = ParetoOptimization(obj_size=obj)
archive_overlap = 2

for i, problem in enumerate(problems):
    comparing.append(['NSGA3_6', 'k_' + str(ks[i]).replace('.', '') + '_l_' + str(ls[i]).replace('.', '')])
    print("******************\n", problem, "\n***********************\n")
    reference_point = pickle.load(
        open('E:\\' + problem + '_' + str(obj) + '_reference_point.pickle', 'rb'))
    file_regex = r'NSGA3_6(.*)' + problem + r'_(.*)' + str(obj) + r'_objectives_'
    stored_files = MultiFileReader(file_regex,
                                   dir="C:\\Users\\amy_l\\PycharmProjects\\FEA\\results\\factorarchive\\")
    experiment_filenames = stored_files.path_to_files
    # for j, compare in enumerate(comparing):
    experiments = [x for x in experiment_filenames if comparing[i][0] in x and problem in x]
    print(experiments)
    for k, l in zip(ks, ls):
        print('k: ', k, 'l: ', l)
        avg_hv = {'nsga3':[], 'FAM':[]}
        avg_div = {'nsga3':[], 'FAM':[]}
        for id,experiment in enumerate(experiments):
            try:
                results = pickle.load(open(experiment, 'rb'))
            except EOFError:
                print('issues with file: ', experiment)
                continue
            if isinstance(results, Result):
                fa = FactorArchive(obj, dimensions=100, percent_best=k, percent_remaining=l)
                fa.update_archive(results)
                overlapping_archive = fa.find_archive_overlap(nr_archives_overlapping=archive_overlap)
                print(len(overlapping_archive))
            else:
                continue
            if len(overlapping_archive) > 2:
                sol_set = environmental_solution_selection_nsga2(results.flatten_archive(),
                                                                     sol_size=len(overlapping_archive))
                overlapping_fitness = np.array([np.array(sol.fitness) / reference_point for sol in overlapping_archive])
                overlapping_hv = wfg(overlapping_fitness, np.ones(obj))
                nsga3_hv = wfg(results.F, np.ones(obj))
                overlapping_div = po.calculate_diversity(overlapping_fitness,
                                                         normalized=True)
                nsga3_div = po.calculate_diversity(results.F, normalized=True)
                avg_hv['nsga3'].append(nsga3_hv)
                avg_hv['FAM'].append(overlapping_hv)
                avg_div['nsga3'].append(nsga3_div)
                avg_div['FAM'].append(overlapping_div)
            else:
                fig = plt.figure()
                ax = fig.add_subplot(projection='polar')
                sol = overlapping_archive[0]
                solution = np.array([x for x in sol.fitness / reference_point])
                max_value = np.max(solution)
                obj_dict = dict()
                for i, fitness in enumerate(solution):
                    keystring = "Objective " + str(i + 1)
                    obj_dict[keystring] = [fitness]
                df = pd.DataFrame(obj_dict)

                # calculate values at different angles
                z = df.rename(index={0: 'value'}).T.reset_index()
                z = z.append(z.iloc[0], ignore_index=True)
                z = z.reindex(np.arange(z.index.min(), z.index.max() + 1e-10, 0.05))

                z['angle'] = np.linspace(0, 2 * np.pi, len(z))
                z.plot.scatter('angle', 'value', ax=ax, legend=False)
                z['value'] = z['value'].interpolate(method='linear')

                z.plot('angle', 'value', ax=ax, legend=False)
                ax.fill_between(z['angle'], 0, z['value'], alpha=0.1)
                ax.set_xticks(z.dropna()['angle'].iloc[:-1])
                ax.set_xticklabels(z.dropna()['index'].iloc[:-1])
                ax.set_xlabel(None)
                ax.set_ylabel(None)
                tickvalues = [round(j * max_value / 5, 3) for j in range(5)]
                ax.set_yticks(tickvalues)
                ax.set_rlabel_position(-65)

                # striped background
                n = 5
                vmax = max_value
                for i in np.arange(n):
                    ax.fill_between(
                        np.linspace(0, 2 * np.pi, 100), vmax / n / 2 * i * 2, vmax / n / 2 * (i * 2 + 1),
                        color='silver', alpha=0.1)
                plt.show()
                pathtosave = "./objective_polarplots/" + problem + "/NGSA3/"
                fig.savefig(pathtosave+problem + "_" + str(obj) + "-obj_NSGA3_OAM-S_"+str(id))
        print("nsga3 FAM: ", avg_hv['FAM'], avg_div['FAM'])
