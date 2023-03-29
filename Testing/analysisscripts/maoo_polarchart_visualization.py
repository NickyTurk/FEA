import os
import pickle
import random
import re

from pymoo.core.result import Result
from shortuuid import uuid
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from MOO.MOEA import MOEA
from MOO.archivemanagement import FactorArchive, environmental_solution_selection_nsga2
from utilities.multifilereader import MultiFileReader

problem = "DTLZ5"
obj = 5
algorithm = "NSGA2_FactorArchive_k_05_l_02" #"NSGA2_FactorArchive_k_04_l_03"
get_ES_plot = False
external_bool = False
arch_overlap = 2

full_name = problem + "_" + str(obj) + "-obj" + "_" + algorithm + "_"

if external_bool:
    archive_type = "external_archive"
    if get_ES_plot:
        full_name = full_name + "ES-E_"
    else:
        full_name = full_name + "OAM-E_"
else:
    archive_type = "single_archive"
    if get_ES_plot:
        full_name = full_name + "ES-S_"
    else:
        full_name = full_name + "OAM-S_"

print("******************\n",problem,"\n***********************\n")
reference_point = pickle.load(
    open('E:\\reference_points\\' + problem + '_' + str(obj) + '_reference_point.pickle', 'rb'))
file_regex = algorithm + r'(.*)' + problem + r'_(.*)' + str(obj) + r'_objectives_'
stored_files = MultiFileReader(file_regex,
                               dir="C:\\Users\\amy_l\\PycharmProjects\\FEA\\results\\factorarchive\\" + problem + "\\")
experiment_filenames = stored_files.path_to_files
for expid,experiment in enumerate(experiment_filenames):
    fig = plt.figure()
    og_archive = pickle.load(open(experiment, 'rb'))
    if isinstance(og_archive, FactorArchive):
        if not get_ES_plot:
            archive = og_archive.find_archive_overlap(nr_archives_overlapping=arch_overlap)
        else:
            continue
    elif isinstance(og_archive, MOEA):
        if external_bool:
            fa = og_archive.nondom_archive
        else:
            fa = FactorArchive(obj,100, percent_best=.4, percent_remaining=.3)
            fa.update_archive(og_archive.nondom_pop)
        temp_archive = fa.find_archive_overlap(nr_archives_overlapping=arch_overlap)
        if get_ES_plot:
            sol_len = len(temp_archive)
            if external_bool:
                archive = environmental_solution_selection_nsga2(og_archive.nondom_archive.flatten_archive(), sol_len)
            else:
                archive = environmental_solution_selection_nsga2(og_archive.nondom_pop, sol_len)
        else:
            archive = temp_archive
    else:
        if isinstance(og_archive, Result):
            archive = og_archive.F
        else:
            archive = og_archive
    if len(archive) > 0:
        print(archive)
        ax = fig.add_subplot(projection='polar')
        try:
            max_value = np.max(np.array([x.fitness/reference_point for x in archive]))
        except AttributeError:
            max_value = np.max(np.array([x / reference_point for x in archive]))
        for sol in archive:
            try:
                solution = np.array([x for x in sol.fitness/reference_point])
            except AttributeError:
                solution = np.array([x for x in sol/reference_point])
            obj_dict = dict()
            for i, fitness in enumerate(solution):
                keystring = "Objective "+str(i+1)
                obj_dict[keystring] = [fitness]
            df = pd.DataFrame(obj_dict)

            # calculate values at different angles
            z = df.rename(index={0: 'value'}).T.reset_index()
            z = z.append(z.iloc[0], ignore_index=True) #pd.concat([z, z.iloc[0]], ignore_index=True)
            z = z.reindex(np.arange(z.index.min(), z.index.max() + 1e-10, 0.05))

            z['angle'] = np.linspace(0, 2*np.pi, len(z))
            z.plot.scatter('angle', 'value', ax=ax, legend=False)
            z['value'] = z['value'].interpolate(method='linear') #method='linear'

            # plot
            # fig = plt.figure()
            # ax = fig.add_subplot(projection='polar')
            z.plot('angle', 'value', ax=ax, legend=False)
            # ax.fill_between(z['angle'], 0, z['value'], alpha=0.1)
            ax.set_xticks(z.dropna()['angle'].iloc[:-1])
            ax.set_xticklabels(z.dropna()['index'].iloc[:-1])
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            tickvalues = [round(j*max_value/5,3) for j in range(5)]
            ax.set_yticks(tickvalues)
            ax.set_rlabel_position(-65)

        # striped background
        n = 5
        vmax = max_value
        for i in np.arange(n):
            ax.fill_between(
                np.linspace(0, 2*np.pi, 100), vmax/n/2*i*2, vmax/n/2*(i*2+1),
                color='silver', alpha=0.1)
        plt.show()
        plt.close()
        # if not os.path.isdir("./objective_polarplots/"+problem+"/"+archive_type+"/"):
        #     os.mkdir("./objective_polarplots/"+problem+"/"+archive_type+"")
        #     if not os.path.isdir("./objective_polarplots/"+problem+"/"+archive_type+"/"+algorithm):
        #         os.mkdir("./objective_polarplots/"+problem+"/"+archive_type+"/"+algorithm)
        # if not os.path.isdir("./objective_polarplots/" + problem + "/"+archive_type+"/environmental_selection/"):
        #     os.mkdir("./objective_polarplots/" + problem + "/"+archive_type+"/environmental_selection/")
        # if "NSGA3" in algorithm:
        #     pathtosave="./objective_polarplots/"+problem+"/NGSA3/"
        # elif algorithm == "NSGA2":
        #     regex = r'FactorArchive_k_([0-9]*)_l_([0-9]*)'
        #     filename = re.match(regex, experiment)
        #     print(filename)
        #     pathtosave = "./objective_polarplots/" + problem + "/" + algorithm + "/" + filename[0] +"/"
        # else:
        #     if get_ES_plot:
        #         pathtosave = "./objective_polarplots/" + problem + "/"+archive_type+"/environmental_selection/"
        #     else:
        #         pathtosave = "./objective_polarplots/" + problem + "/"+archive_type+"/" + algorithm + "/"
        # if not os.path.isdir(pathtosave):
        #     os.mkdir(pathtosave)
        # fig.savefig(pathtosave+full_name+"_"+str(expid)+"_"+str(arch_overlap)+"_"+uuid())
