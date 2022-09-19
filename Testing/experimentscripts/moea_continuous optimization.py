from datetime import timedelta

from pymoo.decomposition.tchebicheff import Tchebicheff
from pymoo.factory import get_problem, get_reference_directions

from MOO.MOEA import MOEA, NSGA2, SPEA2, MOEAD
from utilities.util import *

import os, re, time, pickle
from pymoo.problems.many.dtlz import DTLZ1

dimensions = 100
ga_run = 100
population = 500
nr_obj = 3

ref_dirs = get_reference_directions("das-dennis", nr_obj, n_partitions=12)
pf = get_problem("dtlz1", n_var=dimensions, n_obj=nr_obj).pareto_front(ref_dirs)
reference_point = np.max(pf, axis=0)

current_working_dir = os.getcwd()
path = re.search(r'^(.*?[\\/]FEA)', current_working_dir)
path = path.group()

# moea1 = partial(SPEA2, population_size=population, ea_runs=ga_run)
# moea2 = partial(NSGA2, population_size=population, ea_runs=ga_run)
# moea3 = partial(MOEAD, ea_runs=ga_run, weight_vector=ref_dirs, n_neighbors=10, problem_decomposition=Tchebicheff())

name = 'SPEA2'


@add_method(MOEA)
def calc_fitness(variables, gs=None, factor=None):
    dtlz = get_problem("dtlz1", n_var=dimensions, n_obj=nr_obj)
    objective_values = dtlz.evaluate(variables)
    return tuple(objective_values)


for i in range(5):
    print('##############################################\n', i)
    start = time.time()
    filename = path + '/results/DTLZ1/' + name + '/' + name + '_DTLZ1_' + str(nr_obj) + \
        '_objectives_ea_runs_' + str(ga_run) + '_population_' + str(population) + time.strftime(
        '_%d%m%H%M%S') + '.pickle'
    moo = SPEA2(dimensions=dimensions, value_range=[0.0, 1.0], reference_point=[1.0,1.0,1.0],
                   ea_runs=ga_run)
    moo.run()
    end = time.time()
    file = open(filename, "wb")
    pickle.dump(moo, file)
    elapsed = end - start
    print(
        "Alg with %d runs and population %d took %s" % (ga_run, population, str(timedelta(seconds=elapsed))))
