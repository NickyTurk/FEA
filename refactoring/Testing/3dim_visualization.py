from refactoring.utilities.multifilereader import MultiFileReader
import pickle, random
import numpy as np
from pymoo.util.nds.non_dominated_sorting import find_non_dominated
from itertools import combinations
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from IPython.core.display import display

nondom_solutions = dict()
lengths = dict()
comparing = ['population_500', 'grouping_100_100', 'grouping_100_80', 'grouping_200_200', 'grouping_200_160']
label_names = ['NSGA-II', 'CC-NSGA-II-100', 'F-NSGA-II-100', 'CC-NSGA-II-200', 'F-NSGA-II-200']
total_front = []

file_regex = r'_single_knapsack_3_objectives_'
stored_files = MultiFileReader(file_regex)
experiment_filenames = stored_files.path_to_files

fig = plt.figure()
ax = plt.axes(projection='3d')
plt.ion()
# ax.axes.set_xlim3d(left=600, right=1000)
# ax.axes.set_ylim3d(bottom=14000, top=20000)
# ax.axes.set_zlim3d(bottom=1400, top=1800)
colors = ['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000']
markers = ['o', 'x', '*', 's', 'd']
lines = []
for i, compare in enumerate(comparing):
    experiment = [x for x in experiment_filenames if compare+'_' in x]
    if experiment:
        rand_int = random.randint(0, len(experiment)-1)
        print(rand_int)
        experiment = experiment[rand_int]
    else:
        break
    feamoo = pickle.load(open(experiment, 'rb'))
    solutions = np.array([np.array(x.fitness) for x in feamoo.nondom_archive])
    ax.scatter3D(solutions[:,0], solutions[:,1], solutions[:,2], colors[i], marker=markers[i])
    lines.append(mlines.Line2D([], [], color=colors[i], marker=markers[i],
                              markersize=15, label=label_names[i]))
ax.legend(handles=lines)
plt.show()
