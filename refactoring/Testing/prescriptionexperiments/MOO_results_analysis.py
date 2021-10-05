import pickle, glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

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
iterations = []
alg_stats = dict()
for field_name in field_names:
    alg_stats[field_name] = dict()
    for method in methods:
        print(method)
        alg_stats[field_name][method] = dict()
        experiment = [x for x in experiment_filenames if method+'_' in x and field_name in x.lower()]
        print(experiment)
        if experiment:
            experiment = experiment[0]
        else:
            break
        feamoo = pickle.load(open(experiment, 'rb'))
        print(feamoo.iteration_stats[-1])
        if method != 'NSGA2':
            alg_stats[field_name][method]['iterations'] = [stats['FEA_run'] for i, stats in
                                                           enumerate(feamoo.iteration_stats) if i != 0]
        else:
            alg_stats[field_name][method]['iterations'] = [stats['GA_run'] for i, stats in
                                                           enumerate(feamoo.iteration_stats) if i != 0]
        alg_stats[field_name][method]['diversity'] = [stats['diversity'] for i, stats in
                                                      enumerate(feamoo.iteration_stats) if i != 0]
        alg_stats[field_name][method]['hv'] = [stats['hypervolume'] for i, stats in enumerate(feamoo.iteration_stats) if
                                               i != 0]
        alg_stats[field_name][method]['nd'] = [stats['ND_size'] for i, stats in enumerate(feamoo.iteration_stats) if
                                               i != 0]

stat_names = ['hv', 'diversity', 'nd']

stat_name = 'hv'
ccea = list(alg_stats['henrys']['CCEAMOO'][stat_name])
nsga = list(alg_stats['henrys']['NSGA2'][stat_name])
fea = list(alg_stats['henrys']['FEAMOO'][stat_name])

fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

axs[0].plot(alg_stats['henrys']['CCEAMOO']['iterations'], ccea, color='tab:blue', marker='.')
axs[0].plot(alg_stats['henrys']['NSGA2']['iterations'], nsga, color='#2ca02c', marker="^")
axs[0].plot(alg_stats['henrys']['FEAMOO']['iterations'], fea, color='tab:red', marker="*")
axs[0].set_title("Henrys")

ccea = list(alg_stats['sec35middle']['CCEAMOO'][stat_name])
nsga = list(alg_stats['sec35middle']['NSGA2'][stat_name])
fea = list(alg_stats['sec35middle']['FEAMOO'][stat_name])

axs[1].plot(alg_stats['sec35middle']['CCEAMOO']['iterations'], ccea, color='tab:blue',marker='.')
axs[1].plot(alg_stats['sec35middle']['NSGA2']['iterations'], nsga, color='#2ca02c', marker="^")
axs[1].plot(alg_stats['sec35middle']['FEAMOO']['iterations'], fea, color='tab:red', marker="*")
axs[1].set_title("Sec35Mid")
axs[1].set_xlabel("Generations")

ccea = list(alg_stats['sec35west']['CCEAMOO'][stat_name])
nsga = list(alg_stats['sec35west']['NSGA2'][stat_name])
fea = list(alg_stats['sec35west']['FEAMOO'][stat_name])

axs[2].plot(alg_stats['sec35west']['CCEAMOO']['iterations'], ccea, color='tab:blue',marker='.')
axs[2].plot(alg_stats['sec35west']['NSGA2']['iterations'], nsga, color='#2ca02c', marker="^")
axs[2].plot(alg_stats['sec35west']['FEAMOO']['iterations'], fea, color='tab:red', marker="*")
axs[2].set_title("Sec35West")
fig.suptitle('')

ccea = mlines.Line2D([], [], color='#1f77b4', marker='.',
                     markersize=12, label='CC-NSGA-II')
nsga = mlines.Line2D([], [], color='#2ca02c', marker='^',
                     markersize=12, label='NSGA-II')
fea = mlines.Line2D([], [], color='tab:red', marker='*',
                    markersize=12, label='F-NSGA-II')
axs[1].legend(handles=[nsga, ccea, fea], bbox_to_anchor=(0, 1.15, 1., .105), loc='center',
              ncol=3, mode="expand", borderaxespad=0.)
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.suptitle('Hypervolume', size='18')
fig.tight_layout(pad=1)
plt.show()

# stat_name = 'diversity'
# ccea = list(alg_stats['henrys']['CCEAMOO'][stat_name])
# nsga = list(alg_stats['henrys']['NSGA2'][stat_name])
# fea = list(alg_stats['henrys']['FEAMOO'][stat_name])
# data = [nsga, ccea, fea]
# fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
# axs[0].set_ylabel('')
# axs[0].boxplot(data)
# axs[0].set_title("Henrys")
# axs[0].set_xticklabels(['NSGA-II', 'CC-NSGA-II', 'F-NSGA-II'])
#
# ccea = list(alg_stats['sec35middle']['CCEAMOO'][stat_name])
# nsga = list(alg_stats['sec35middle']['NSGA2'][stat_name])
# fea = list(alg_stats['sec35middle']['FEAMOO'][stat_name])
# data = [nsga, ccea, fea]
# axs[1].boxplot(data)
# axs[1].set_title("Sec35Mid")
# axs[1].set_xticklabels(['NSGA-II', 'CC-NSGA-II', 'F-NSGA-II'])
#
# ccea = list(alg_stats['sec35west']['CCEAMOO'][stat_name])
# nsga = list(alg_stats['sec35west']['NSGA2'][stat_name])
# fea = list(alg_stats['sec35west']['FEAMOO'][stat_name])
# data = [nsga, ccea, fea]
# axs[2].boxplot(data)
# axs[2].set_title("Sec35West")
# axs[2].set_xticklabels(['NSGA-II', 'CC-NSGA-II', 'F-NSGA-II'])
#
# fig.suptitle('')
# plt.suptitle('Spread Indicator', size='18')
# fig.tight_layout(pad=1)
# plt.show()


