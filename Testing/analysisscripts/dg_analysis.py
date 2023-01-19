import pickle
import re

import numpy as np

from FEA.factorarchitecture import FactorArchitecture
from utilities.multifilereader import MultiFileReader

problems = ['DTLZ1', 'DTLZ2', 'DTLZ3', 'DTLZ4', 'DTLZ5', 'DTLZ6']
nr_objs = [10]
for problem in problems:
    for n_obj in nr_objs:
        file_regex = "DG_"+problem+"_"+str(n_obj)+r'_eps(.*)'
        stored_files = MultiFileReader(file_regex=file_regex, dir="D:\\factor_architecture_files\\DG_MOO\\")
        for file in stored_files.path_to_files:
            print(file)
            fa = FactorArchitecture(1000)
            fa.load_architecture(file)
            print("number of factors: ", len(fa.factors))
            print("length of each factor: ", [len(fac) for fac in fa.factors])
            overlap_sizes = []
            for i, fac in enumerate(fa.factors):
                for j in range(i+1,len(fa.factors)):
                    overlap = np.intersect1d(fac, fa.factors[j])
                    if len(overlap) > 0:
                        overlap_sizes.append(len(overlap))
            print( "Number of overlaps: ", len(overlap_sizes), ". Length of overlaps: ", overlap_sizes)
            print("average overlap size: ", np.mean(overlap_sizes))
            print("average length of factors: ", np.mean([len(fac) for fac in fa.factors]))


#
# problems = ['DTLZ1', 'DTLZ2', 'DTLZ3', 'DTLZ4', 'DTLZ5', 'DTLZ6']
# algorithms = ['NSGA2']  # , 'SPEA2', 'MOEAD', 'MOEADPBI']
# nr_objs = [3,5,10]
# decompositions = ['diff_grouping_MOO']
#
# for n_obj in nr_objs:
#     for problem in problems:
#         for alg in algorithms:
#             for decomp in decompositions:
#                 if decomp == "linear_100_100" or decomp == "classic_random_100":
#                     name = "CC" + alg
#                 elif decomp == "population_500":
#                     name = alg
#                 else:
#                     name = "F" + alg
#                 file_regex = name + r'_' + problem + r'_(.*)' + re.escape(str(n_obj)) + r'_objectives_(.*)' + decomp
#                 print('************************************************\n', name, problem, str(n_obj))
#                 stored_files = MultiFileReader(file_regex=file_regex, dir="D:/" + problem + "/" + name + "/")
#                 file_list = stored_files.path_to_files
#                 if len(file_list) != 0:
#                     file = file_list[0]
#                     try:
#                         alg_object = pickle.load(open(file, 'rb'))
#                     except EOFError:
#                         print("error in file: ", file)
#                     if alg == 'MOEAD':
#                         arch = np.array([np.array(sol.fitness) for sol in alg_object.nondom_archive])
#                         if np.any(arch < 0):
#                             continue
#                     fa = alg_object.factor_architecture
#                     print("number of factors: ", len(fa.factors))
#                     print("length of each factor: ", [len(fac) for fac in fa.factors])
#                     overlap_sizes = []
#                     for i, fac in enumerate(fa.factors):
#                         for j in range(i+1,len(fa.factors)):
#                             overlap = np.intersect1d(fac, fa.factors[j])
#                             if len(overlap) > 0:
#                                 overlap_sizes.append(len(overlap))
#                     print( "Number of overlaps: ", len(overlap_sizes), ". Length of overlaps: ", overlap_sizes)
#                     print("average overlap size: ", np.mean(overlap_sizes))
#                     print("average length of factors: ", np.mean([len(fac) for fac in fa.factors]))