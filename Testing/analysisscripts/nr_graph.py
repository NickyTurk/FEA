import csv
from operator import attrgetter
from pyproj import Transformer
from predictionalgorithms.yieldprediction import create_indexed_dataframe
from pymoo.util.nds.non_dominated_sorting import find_non_dominated
from optimizationproblems.prescription import Prescription
from utilities.multifilereader import MultiFileReader

try:
    import _pickle as pickle
except:
    import pickle
import re, os
import pandas as pd
import numpy as np

current_working_dir = os.getcwd()
path = re.search(r'^(.*?[\\/]FEA)',current_working_dir)
path = path.group()

field_files= [path+"/utilities/saved_fields/Henrys.pickle"]
field_names = ['Henrys']
methods = ["CCEAMOO", "NSGA", "FEAMOO", "COMBINED"]
objectives = ['jumps', 'fertilizer_rate', 'NR']

combined_front = []
all_data = dict() #{'henrys': { 'cceamoo':{'jumps': 0, 'strat': 0, 'fertilizer_rate': 0, 'center': 0}, 'nsga2': {} } }
for fieldfile, field_name in zip(field_files, field_names):
    field = pickle.load(open(fieldfile, 'rb'))
    all_data[field_name] = dict()
    for method in methods:
        all_data[field_name][method] = dict()
        if method != 'COMBINED':
            to_find = method+'_'+field_name
            mf = MultiFileReader(to_find, dir=path+'/results/prescriptions/optimized/')
            experiment_filenames = mf.path_to_files
            print(experiment_filenames)
            nondom_archive = []
            find_center_obj = []
            for experiment in experiment_filenames:
                feamoo = pickle.load(open(experiment, 'rb'))
                # print(feamoo.iteration_stats[-1])
                nondom_archive.extend(feamoo.nondom_archive)
            try:
                fitnesses = np.array([np.array(x.objective_values) for x in nondom_archive])
            except ValueError:
                fitnesses = np.array([np.array(x.fitness) for x in nondom_archive])
            nondom_indeces = find_non_dominated(fitnesses)
            nondom_archive = [fitnesses[j] for j in nondom_indeces]
            combined_front.extend(nondom_archive)
            for i,obj in enumerate(objectives):
                nondom_archive.sort(key=lambda test_list: test_list[i])
                prescription = nondom_archive[0]
                find_center_obj.append(np.array(prescription))
                all_data[field_name][method][obj] = prescription[-1]
            find_center_obj = np.array(find_center_obj)
            length = find_center_obj.shape[0]
            sum_x = np.sum(find_center_obj[:, 0])
            sum_y = np.sum(find_center_obj[:, 1])
            sum_z = np.sum(find_center_obj[:, 2])
            point = np.array([sum_x / length, sum_y / length, sum_z / length])
            dist = np.sum((nondom_archive- point) ** 2, axis=1)
            idx = np.argmin(dist)
            prescription = nondom_archive[idx]
            all_data[field_name][method]['center'] = prescription[-1]
        else:
            find_center_obj = []
            nondom_indeces = find_non_dominated(combined_front)
            nondom_archive = [combined_front[j] for j in nondom_indeces]
            for i,obj in enumerate(objectives):
                nondom_archive.sort(key=lambda test_list: test_list[i])
                prescription = nondom_archive[0]
                find_center_obj.append(np.array(prescription))
                all_data[field_name][method][obj] = prescription[-1]
            find_center_obj = np.array(find_center_obj)
            length = find_center_obj.shape[0]
            sum_x = np.sum(find_center_obj[:, 0])
            sum_y = np.sum(find_center_obj[:, 1])
            sum_z = np.sum(find_center_obj[:, 2])
            point = np.array([sum_x / length, sum_y / length, sum_z / length])
            dist = np.sum((nondom_archive- point) ** 2, axis=1)
            idx = np.argmin(dist)
            prescription = nondom_archive[idx]
            all_data[field_name][method]['center'] = prescription[-1]

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

names = ['center', 'jumps', 'fertilizer_rate', 'NR']

fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
axs[0].set_ylabel('Net Return ($)')

ccea_yield = list(all_data['Henrys']['CCEAMOO'].values())
nsga_yield = list(all_data['Henrys']['NSGA'].values())
fea_yield = list(all_data['Henrys']['FEAMOO'].values())
comb_yield = list(all_data['Henrys']['COMBINED'].values())

axs[0].scatter(names, comb_yield, color='black', marker="s")
axs[0].plot(names, comb_yield, color='black')
axs[0].scatter(names, ccea_yield)
axs[0].plot(names, ccea_yield)
axs[0].scatter(names, nsga_yield, color='#2ca02c', marker="^")
axs[0].plot(names, nsga_yield, color='#2ca02c')
axs[0].scatter(names, fea_yield, color='tab:red', marker="*")
axs[0].plot(names, fea_yield, color='tab:red')
axs[0].set_title("Henrys")

ccea = mlines.Line2D([], [], color='#1f77b4', marker='.',
                          markersize=12, label='CC-NSGA-II')
nsga = mlines.Line2D([], [], color='#2ca02c', marker='^',
                          markersize=12, label='NSGA-II')
fea = mlines.Line2D([], [], color='tab:red', marker='*',
                          markersize=12, label='F-NSGA-II')
comb = mlines.Line2D([], [], color='black', marker='s',
                          markersize=10, label='Union')
axs[1].legend(handles=[nsga, ccea, fea, comb], bbox_to_anchor=(0, 1.15, 1., .105), loc='center',
           ncol=4, mode="expand", borderaxespad=0.)
#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.suptitle('Predicted Net Return', size='18')
fig.tight_layout(pad=1)
plt.show()