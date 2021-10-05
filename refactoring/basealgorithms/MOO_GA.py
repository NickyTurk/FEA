"""
The Genetic Algorithm implementation to maximize stratification while minimizing jumps between consecutive cells.

NSEA2 class -- algorithm instance
"""
import gc

from refactoring.MOO.paretofront import *
from refactoring.basealgorithms.ga import GA
from refactoring.utilities.util import PopulationMember

import numpy as np
from pymoo.algorithms.nsga2 import calc_crowding_distance
from pymoo.util.nds.non_dominated_sorting import find_non_dominated
from pymoo.util.nds.fast_non_dominated_sort import fast_non_dominated_sort

import random


class MOOEA:
    def __init__(self, combinatorial_options, population_size, dimensions):
        self.combinatorial_values = combinatorial_options
        self.population_size = population_size
        self.dimensions = dimensions

    def initialize_population(self, gs=None, factor=None):
        i = 0
        curr_population = []
        initial_solution = []
        while i < self.population_size:
            if self.combinatorial_values:
                new_solution = random.choices(self.combinatorial_values,
                                              k=self.dimensions)  # Prescription(field=field, index=i, gs=self.global_solution, factor=self.factor)
            else:
                new_solution = [random.random() for x in range(self.dimensions)]
            fitness = self.calc_fitness(new_solution, gs=gs, factor=factor)
            curr_population.append(PopulationMember(new_solution, fitness))
            if i == 0:
                initial_solution = new_solution
            i = i + 1
        return curr_population, initial_solution

    def calc_fitness(self, variables, gs=None, factor=None):
        pass


class NSGA2(MOOEA):
    def __init__(self, evolutionary_algorithm=None, dimensions=100, population_size=500, ea_runs=100,
                 # data_distribution=False,
                 combinatorial_values=[], factor=None, global_solution=None, ref_point = None):
        """
        :param: evolutionary_algorithm -- currently only GA class exists: from refactoring.basealgorithms.ga import GA
        """
        super().__init__(combinatorial_values, population_size, dimensions)
        self.dimensions = dimensions
        self.population_size = population_size
        self.ea = GA(dimensions=dimensions)
        self.curr_population = []
        self.initial_solution = []
        self.factor = factor
        self.global_solution = global_solution
        self.ea_runs = ea_runs
        # self.cell_distribution = data_distribution  # bins are equally distributed across cells: True or False
        self.nondom_pop = []
        self.nondom_archive = []
        self.iteration_stats = []
        self.worst_fitness_ref = ref_point
        self.worst_index = None

    def replace_worst_solution(self, gs):
        """
        :param gs: global solution
        After FEA finishes competition, the global solution replaces the worst solution in each subpopulation
        """
        #nondom_indeces = find_non_dominated(np.array([np.array(x.fitness) for x in self.curr_population]))
        #self.nondom_pop = [self.curr_population[i] for i in nondom_indeces]
        # dominated = [x for x in self.curr_population if x not in self.nondom_pop]
        # if len(dominated) != 0:
        #     diverse_sort = self.diversity_sort(dominated)
        # else:
        #     diverse_sort = self.diversity_sort(self.nondom_pop)
        # worst = diverse_sort[-1]
        if self.worst_index is None:
            diverse_sort = self.diversity_sort(self.curr_population)
            self.worst_index = [i for i, x in enumerate(self.curr_population) if x.fitness == diverse_sort[-1].fitness]
        print('worst index found: ', self.worst_index)
        #idx = np.where(np.array(self.curr_population) == self.worst)
        self.curr_population[self.worst_index[0]] = [gs.variables[i] for i in self.factor]
        # worst_solution_vars = [x for x in self.global_solution.variables]
        # for i, x in zip(self.factor, self.worst.variables):
        #     worst_solution_vars[i] = x
        # return worst_solution_vars

    def select_new_generation(self, i):
        # check for non-domination, then sort based on crowding distance
        total_population = [x for x in self.curr_population]
        fitnesses = np.array([np.array(x.fitness) for x in total_population])
        nondom_indeces = find_non_dominated(fitnesses)
        self.nondom_pop = [total_population[i] for i in nondom_indeces]
        self.nondom_archive.extend(self.nondom_pop)
        if len(self.nondom_pop) < self.population_size:
            new_population = []
            fronts = fast_non_dominated_sort(fitnesses)
            last_front = []
            n_ranked = 0
            for front in fronts:
                # increment the n_ranked solution counter
                n_ranked += len(front)
                # stop if more than this solutions are n_ranked
                if n_ranked >= self.population_size:
                    last_front = front
                    break
                new_population.extend([total_population[i] for i in front])

            sorted_population = self.diversity_sort([total_population[i] for i in last_front])
            length_to_add = self.population_size - len(new_population)
            new_population.extend(sorted_population[:length_to_add])
            self.curr_population = new_population
            worst_fitness = tuple([x for x in self.curr_population[-1].fitness])
        else:
            sorted_population = self.diversity_sort(self.nondom_pop)
            self.curr_population = sorted_population[:self.population_size]
            worst_fitness = tuple([x for x in self.curr_population[-1].fitness])
        random.shuffle(self.curr_population)
        if i == self.ea_runs-1:
            print(worst_fitness)
            self.worst_index = [i for i, x in enumerate(self.curr_population) if x.fitness == worst_fitness]  # np.where(np.array(self.curr_population, dtype=object) == worst)
            print(self.worst_index)

    def diversity_sort(self, population):
        fitnesses = np.array([np.array(x.fitness) for x in population])
        distances = calc_crowding_distance(fitnesses)
        return [x for y, x in sorted(zip(distances, population))]

    def run(self, progressbar=None, writer=None):
        """
        Run the entire non-dominated sorting EA
        """
        self.curr_population, self.initial_solution = self.initialize_population(gs=self.global_solution,
                                                                                 factor=self.factor)
        i = 1
        change_in_nondom_size = []
        old_archive_length = 0
        '''
        Convergence criterion based on change in non-dominated solution set size
        '''
        while i != self.ea_runs: #and len(change_in_nondom_size) < 10:
            children = self.ea.create_offspring(self.curr_population)
            #[print(x) for x in children]
            self.curr_population.extend(
                [PopulationMember(c, self.calc_fitness(c, self.global_solution, self.factor)) for c in children])
            self.select_new_generation(i)
            if self.factor is None and i != 1:
                print("Only entered when running single population NSGA")
                archive_nondom_indeces = find_non_dominated(
                    np.array([np.array(x.fitness) for x in self.nondom_archive]))
                seen = set()
                nondom_archive = [self.nondom_archive[i] for i in archive_nondom_indeces]
                self.nondom_archive = [seen.add(s.fitness) or s for s in nondom_archive if s.fitness not in seen]
                print(len(self.nondom_archive))
                print(seen)
                if len(self.nondom_archive) == old_archive_length and len(self.nondom_archive) >= 10:
                    change_in_nondom_size.append(True)
                else:
                    change_in_nondom_size = []
                old_archive_length = len(self.nondom_archive)

                po = ParetoOptimization()
                eval_dict = po.evaluate_solution(self.nondom_archive, self.worst_fitness_ref)
                eval_dict['GA_run'] = i
                eval_dict['ND_size'] = len(self.nondom_archive)
                self.iteration_stats.append(eval_dict)
                print(eval_dict)
            i += 1


class SPEA2(MOOEA):
    def __init__(self, evolutionary_algorithm, dimensions=100, population_size=200, ea_runs=100,
                 # data_distribution=False,
                 combinatorial_values=[], factor=None, global_solution=None):
        super().__init__(combinatorial_values, population_size, dimensions)
        self.dimensions = dimensions
        self.population_size = population_size
        self.ea = evolutionary_algorithm(dimensions, population_size)
        self.curr_population, self.initial_solution = self.initialize_population(gs=global_solution, factor=factor)
        self.factor = factor
        self.global_solution = global_solution
        self.ea_runs = ea_runs
        self.nondom_pop = []
        self.nondom_archive = []
        self.iteration_stats = []


class MOEAD(MOOEA):
    def __init__(self, evolutionary_algorithm, dimensions=100, population_size=200, ea_runs=100,
                 # data_distribution=False,
                 combinatorial_values=[], factor=None, global_solution=None):
        super().__init__(combinatorial_values, population_size, dimensions)
        self.dimensions = dimensions
        self.population_size = population_size
        self.ea = evolutionary_algorithm(dimensions, population_size)
        self.curr_population, self.initial_solution = self.initialize_population(gs=global_solution, factor=factor)
        self.factor = factor
        self.global_solution = global_solution
        self.ea_runs = ea_runs
        self.nondom_pop = []
        self.nondom_archive = []
        self.iteration_stats = []


class HYPE(MOOEA):
    def __init__(self, evolutionary_algorithm, dimensions=100, population_size=200, ea_runs=100,
                 # data_distribution=False,
                 combinatorial_values=[], factor=None, global_solution=None):
        super().__init__(combinatorial_values, population_size, dimensions)
        self.dimensions = dimensions
        self.population_size = population_size
        self.ea = evolutionary_algorithm(dimensions, population_size)
        self.curr_population, self.initial_solution = self.initialize_population(gs=global_solution, factor=factor)
        self.factor = factor
        self.global_solution = global_solution
        self.ea_runs = ea_runs
        self.nondom_pop = []
        self.nondom_archive = []
        self.iteration_stats = []
