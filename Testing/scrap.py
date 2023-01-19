import pickle, re, os
from MOO.paretofrontevaluation import ParetoOptimization
from utilities.multifilereader import MultiFileReader
from utilities.util import PopulationMember
import numpy as np

problems = ['WFG8']
nr_objs = [5, 10]

for n_obj in nr_objs:
    print(n_obj)
    for problem in problems:
        print(problem)
        stored_files = MultiFileReader(file_regex=r'(.*)' + re.escape(str(n_obj)) + r'_objectives_(.*)',
                                       dir="D:\\" + problem)
        file_list = stored_files.path_to_files
        max_objs = np.zeros(n_obj)
        for file in file_list:
            try:
                alg = pickle.load(open(file, 'rb'))
            except EOFError:
                print('file error: ', file)
                continue
            for i in range(n_obj):
                try:
                    max_found = max([x[i] for x in alg.F])
                    print(len(alg.F))
                except AttributeError:
                    max_found = max([x.fitness[i] for x in alg.nondom_archive])
                    print(len(alg.nondom_archive))
                if max_found > max_objs[i]:
                    max_objs[i] = max_found
        pickle.dump(max_objs, open('D:\\'+problem+'\\'+problem+'_'+str(n_obj)+'_reference_point.pickle', 'wb'))