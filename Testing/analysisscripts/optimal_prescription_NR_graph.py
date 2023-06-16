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

"""
Create Net Return graphs for the optimal prescription map experiments.
Uses four different generated prescriptions:
['fertilizer_rate', 'jumps', 'NR', 'center']
Where the "center" map represents the closest solution to the centroid of the three other prescription maps.
"""

current_working_dir = os.getcwd()
path = re.search(r'^(.*?[\\/]FEA)',current_working_dir)
path = path.group()

field_files= [path+"/utilities/saved_fields/Henrys.pickle"]
field_names = ['henrys']
methods = ["CCNSGA2", "SNSGA2", "FNSGA2", "COMBINED"]
objectives = ['jumps', 'fertilizer_rate', 'NR']

combined_front = []
all_data = dict() #{'henrys': { 'cceamoo':{'jumps': 0, 'strat': 0, 'fertilizer_rate': 0, 'center': 0}, 'nsga2': {} } }
fertilizer_maps = dict()

for fieldfile, field_name in zip(field_files, field_names):
    field = pickle.load(open(fieldfile, 'rb'))
    all_data[field_name] = dict()
    fertilizer_maps[field_name] = dict()
    for method in methods:
        all_data[field_name][method] = dict()
        fertilizer_maps[field_name][method] = dict()
        if method != 'COMBINED':
            to_find = method+'_'+field_name
            mf = MultiFileReader(to_find, dir='D:\\Prescriptions\\CNN_optimized\\')
            experiment_filenames = mf.path_to_files
            print('files: ', experiment_filenames)
            nondom_archive = []
            find_center_obj = []
            center_map = []
            for experiment in experiment_filenames:
                alg_object = pickle.load(open(experiment, 'rb'))
                if alg_object.nondom_archive:
                    nondom_archive.extend(alg_object.nondom_archive)
                else:
                    nondom_archive.extend(alg_object.nondom_pop)
                    print(len(alg_object.nondom_pop[0].variables))
            try:
                fitnesses = np.array([np.array(x.objective_values) for x in nondom_archive])
            except AttributeError:
                fitnesses = np.array([np.array(x.fitness) for x in nondom_archive])
            nondom_indeces = find_non_dominated(fitnesses)
            nondom_fitnesses =[fitnesses[j] for j in nondom_indeces]
            nondom_maps = [nondom_archive[j] for j in nondom_indeces]
            combined_front.extend(nondom_maps)
            for i,obj in enumerate(objectives):
                prescription = sorted(zip(nondom_fitnesses, nondom_maps), key=lambda test_list: test_list[0][i])[0]
                find_center_obj.append(np.array(prescription[0]))
                all_data[field_name][method][obj] = prescription[0][-1]
                fertilizer_maps[field_name][method][obj] = prescription[1]
            find_center_obj = np.array(find_center_obj)
            length = find_center_obj.shape[0]
            sum_x = np.sum(find_center_obj[:, 0])
            sum_y = np.sum(find_center_obj[:, 1])
            sum_z = np.sum(find_center_obj[:, 2])
            point = np.array([sum_x / length, sum_y / length, sum_z / length])
            dist = np.sum((nondom_fitnesses- point) ** 2, axis=1)
            idx = np.argmin(dist)
            prescription = nondom_fitnesses[idx]
            all_data[field_name][method]['center'] = prescription[-1]
            fertilizer_maps[field_name][method]['center'] = nondom_maps[idx]
        else:
            find_center_obj = []
            combined_fitnesses = [x.fitness for x in combined_front]
            nondom_indeces = find_non_dominated(np.array(combined_fitnesses))
            nondom_fitnesses = [combined_fitnesses[j] for j in nondom_indeces]
            nondom_maps = [combined_front[j] for j in nondom_indeces]
            for i,obj in enumerate(objectives):
                prescription = sorted(zip(nondom_fitnesses, nondom_maps), key=lambda test_list: test_list[0][i])[0]
                # nondom_archive.sort(key=lambda test_list: test_list[i])
                # prescription = nondom_archive[0]
                find_center_obj.append(np.array(prescription[0]))
                all_data[field_name][method][obj] = prescription[0][-1]
                fertilizer_maps[field_name][method][obj] = nondom_maps[1]
            find_center_obj = np.array(find_center_obj)
            length = find_center_obj.shape[0]
            sum_x = np.sum(find_center_obj[:, 0])
            sum_y = np.sum(find_center_obj[:, 1])
            sum_z = np.sum(find_center_obj[:, 2])
            point = np.array([sum_x / length, sum_y / length, sum_z / length])
            dist = np.sum((nondom_fitnesses - point) ** 2, axis=1)
            idx = np.argmin(dist)
            prescription = nondom_fitnesses[idx]
            all_data[field_name][method]['center'] = prescription[-1]
            fertilizer_maps[field_name][method]['center'] = nondom_maps[idx]

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

names = ['fertilizer_rate', 'jumps', 'NR', 'center']
for field_name, fieldfile in zip(field_names, field_files):
    field = pickle.load(open(fieldfile, 'rb'))
    # print(field.cell_width, field.cell_length_min, field.cell_length_max)
    gridcell_size = field.cell_list[0].true_bounds.area * (6076.1154855643*60 )
    for method in methods:
        for obj in names:
            print(field_name, method, obj)
            total_fertilizer = 0
            # print(len(fertilizer_maps[field_name][method][obj].variables))
            for fertilizer in fertilizer_maps[field_name][method][obj].variables:
                total_fertilizer += fertilizer*0.98
            print('fertilizer: ', total_fertilizer)


fig = plt.figure()
# fig, axs = plt.subplots(1, 1, figsize=(15, 4.5), sharey=True)
plt.ylabel('Net Return ($)')

print(list(all_data['henrys']['CCNSGA2']))
ccea_yield = list(all_data['henrys']['CCNSGA2'].values())
nsga_yield = list(all_data['henrys']['SNSGA2'].values())
fea_yield = list(all_data['henrys']['FNSGA2'].values())
comb_yield = list(all_data['henrys']['COMBINED'].values())

plt.scatter(names, comb_yield, color='black', marker="s")
plt.plot(names, comb_yield, color='black')
plt.scatter(names, ccea_yield)
plt.plot(names, ccea_yield)
plt.scatter(names, nsga_yield, color='#2ca02c', marker="^")
plt.plot(names, nsga_yield, color='#2ca02c')
plt.scatter(names, fea_yield, color='tab:red', marker="*")
plt.plot(names, fea_yield, color='tab:red')

ccea = mlines.Line2D([], [], color='#1f77b4', marker='.',
                          markersize=12, label='CCNSGA2')
nsga = mlines.Line2D([], [], color='#2ca02c', marker='^',
                          markersize=12, label='NSGA2')
fea = mlines.Line2D([], [], color='tab:red', marker='*',
                          markersize=12, label='FNSGA2')
comb = mlines.Line2D([], [], color='black', marker='s',
                          markersize=10, label='Union')
plt.legend(handles=[nsga, ccea, fea, comb],loc='upper center') #bbox_to_anchor=(0, 1.15, 1., .105),
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.suptitle('RF Predicted Net Return\n Henrys', size='18')
fig.tight_layout(pad=1)
plt.show()