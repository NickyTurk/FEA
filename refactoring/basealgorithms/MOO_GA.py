"""
Contains different algorithm implementations of single population multi-objective optimization algorithms.
MOOEA class -- Superclass for all Multi-objective Evolutionary Algorithms.
NSGA2 class -- Inherits from MOOEA. Non-Dominated Sorting Genetic Algorithm II (Deb, et al.)
SPEA2 class -- Strength Pareto Evolutionary Algorithm
HypE class --
MOEAD class --
"""

from pymoo.util.misc import euclidean_distance
from refactoring.MOO.paretofront import *
from refactoring.basealgorithms.ga import GA
from refactoring.utilities.util import PopulationMember, compare_solutions

from pymoo.algorithms.nsga2 import calc_crowding_distance
from pymoo.util.nds.non_dominated_sorting import find_non_dominated
from pymoo.util.nds.fast_non_dominated_sort import fast_non_dominated_sort

from platypus.tools import DistanceMatrix
from platypus.indicators import Hypervolume

import numpy as np
import random


class MOOEA:
    def __init__(self, combinatorial_options, population_size, dimensions):
        """
        Superclass for all MOO EAs.
        @param combinatorial_options: List of values, E.g. [0,1]. Set of values to create solutions from when dealing
                with combinatorial optimization.
        @param population_size: Integer. Number of individuals to form the population in a single generation.
        @param dimensions: Integer. Number of variables in a single individual. When no combinatorial_options are
                defined, these will be floating point numbers between 0.0 and 1.0]
        Contains methods initialize_population(gs, factor) and calc_fitness(variables, gs, factor).
        Where 'gs'(=global solution) and 'factor' are only used when using the MOEA's as a base-algorithm for FEA.
        """
        self.combinatorial_values = combinatorial_options
        self.population_size = population_size
        self.dimensions = dimensions

    def initialize_population(self, gs=None, factor=None):
        """
        @param gs: Global Solution. Only used for FEA base-algorithms. When implementing any MOOEA as the base-algorithm
                for FEA, each algorithm instance will only be optimizing a subpopulation consisting of a subset of the
                variable space. But to evaluate the fitness of the individuals, a global solution is necessary to plug
                in the subset of variables.
        @param factor: Only used for FEA base-algorithms. The factor defines which subset of variables a specific
                subpopulation will be looking at.
        @return: The initial population: A list of PopulationMember(variables=[dimensions], fitness=())) instances.
        """
        i = 0
        curr_population = []
        initial_solution = []
        while i < self.population_size:
            if self.combinatorial_values:
                new_solution = random.choices(self.combinatorial_values,
                                              k=self.dimensions)  # Prescription(field=field, index=i, gs=self.global_solution, factor=self.factor)
            else:
                new_solution = [random.random() for x in range(self.dimensions)]
            curr_population.append(PopulationMember(new_solution, self.calc_fitness(new_solution, gs=gs, factor=factor)))
            if i == 0:
                initial_solution = new_solution
            i = i + 1
        return curr_population, initial_solution

    def calc_fitness(self, variables, gs=None, factor=None):
        """
        An empty function to fill in dynamically based on the problem.
        @param variables: List of variables to evaluate.
        @param gs: Global Solution. Only used for FEA base-algorithms. When implementing any MOOEA as the base-algorithm
                for FEA, each algorithm instance will only be optimizing a subpopulation consisting of a subset of the
                variable space. But to evaluate the fitness of the individuals, a global solution is necessary to plug
                in the subset of variables.
        @param factor: Only used for FEA base-algorithms. The factor defines which subset of variables a specific
                subpopulation will be looking at.
        @return: Tuple of fitness values. The number of fitness values depends on the number of objectives.
        """
        pass


class NSGA2(MOOEA):
    def __init__(self, evolutionary_algorithm=GA, dimensions=100, population_size=500, ea_runs=100,
                 # data_distribution=False,
                 combinatorial_values=[], factor=None, global_solution=None, ref_point = None):
        """

        @param evolutionary_algorithm: The base algorithm to use. Currently only Genetic Algorithm (GA) is implemented.
        @param dimensions: Integer. Number of variables in a single individual.
        @param population_size: Integer. Number of individuals to form the population in a single generation.
        @param ea_runs: Integer. How many generations to run NSGA2 for.
        @param combinatorial_values: List of values, E.g. [0,1]. Set of values to create solutions from when dealing
                with combinatorial optimization. If empty list [], floating point numbers between 0.0 and 1.0 are used.
        @param factor:
        @param global_solution: Global Solution. List of variables at full dimensionality.
                Only used for FEA base-algorithms when implementing any MOOEA as the base-algorithm for FEA.
                To evaluate the fitness of the individuals, a global solution is necessary to plug
                in the subset of variables.
        @param ref_point: List of worst potential fitness values. Has as many fitness values as there are objectives.
                Used to calculate the Hypervolume indicator.
        """
        super().__init__(combinatorial_values, population_size, dimensions)
        self.dimensions = dimensions
        self.population_size = population_size
        self.ea = evolutionary_algorithm(dimensions=dimensions)
        self.curr_population = []
        self.initial_solution = []
        self.factor = factor
        self.global_solution = global_solution
        self.ea_runs = ea_runs
        self.nondom_pop = []
        self.nondom_archive = []
        self.iteration_stats = []
        self.worst_fitness_ref = ref_point
        self.worst_index = None
        self.random_nondom_solutions = []

    def replace_worst_solution(self, gs):
        """

        @param gs: List of Global Solution(s).
                Used to replace the worst solution(s) with.
        Finds the worst solution based on fitness values and replaces with global solution.
        There can be multiple "worst" solutions, in which case as many are replaces as there are global solutions.
        """
        if self.worst_index is None or self.worst_index == []:
            diverse_sort = self.sorting_mechanism(self.curr_population)
            self.worst_index = [i for i, x in enumerate(self.curr_population) if x.fitness == diverse_sort[-1].fitness]
        if len(gs) > len(self.worst_index):
            for i, idx in enumerate(self.worst_index):
                self.curr_population[idx] = PopulationMember(variables=[gs[i].variables[j] for j in self.factor], fitness=gs[i].fitness)
        elif len(self.worst_index) > len(gs):
            for i, sol in enumerate(gs):
                self.curr_population[self.worst_index[i]] = PopulationMember(variables=[sol.variables[j] for j in self.factor], fitness=sol.fitness)
        elif len(self.worst_index) == len(gs):
            for idx, sol in zip (self.worst_index, gs):
                self.curr_population[idx] = PopulationMember(variables=[sol.variables[j] for j in self.factor], fitness=sol.fitness)

    def select_new_generation(self, i):
        """
        After the base-algorithm (e.g. GA) has created the offspring:
        1. check for non-domination
        2. sort based on crowding distance
        3. select next generation based on sorted population
        @param i: Which generation is the algorithm on.
                If it is on the last generation, calculate which individual has the worst fitness.
        @return: shuffled new population
        """
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

            sorted_population = self.sorting_mechanism([total_population[i] for i in last_front])
            length_to_add = self.population_size - len(new_population)
            new_population.extend(sorted_population[:length_to_add])
            self.curr_population = new_population
            worst_fitness = tuple([x for x in self.curr_population[-1].fitness])
        else:
            sorted_population = self.sorting_mechanism(self.nondom_pop)
            self.curr_population = sorted_population[:self.population_size]
            worst_fitness = tuple([x for x in self.curr_population[-1].fitness])
        random.shuffle(self.curr_population)
        if i == self.ea_runs-1 and self.factor is not None:
            #nondom_indeces = find_non_dominated(np.array([np.array(x.fitness) for x in self.nondom_archive]))
            #nondom_archive = [self.nondom[i] for i in nondom_indeces]
            seen = set()
            for s in self.nondom_pop:
                if s.fitness not in seen:
                    seen.add(s.fitness)
                    self.nondom_archive.append(s)
            choice = random.choice(self.nondom_archive)
            full_solution = [x for x in self.global_solution.variables]
            for i, x in zip(self.factor, choice.variables):
                full_solution[i] = x
            self.random_nondom_solutions.append(full_solution)
            self.worst_index = [i for i, x in enumerate(self.curr_population) if x.fitness == worst_fitness]  # np.where(np.array(self.curr_population, dtype=object) == worst)

    def sorting_mechanism(self, population):
        """
        NSGA2 specific operation.
        Sorts solutions based on the crowding distance to maximize diversity.
        @param population: List of individuals to be sorted.
        @return: List of sorted individuals.
        """
        fitnesses = np.array([np.array(x.fitness) for x in population])
        distances = calc_crowding_distance(fitnesses)
        return [x for y, x in sorted(zip(distances, population))]

    def run(self, fea_run=0):
        """
        Run the full algorithm for the set number of 'ea_runs' or until a convergence criterion is met.
        @param fea_run: Which generation the FEA is on, if NSGA2 is being used as the base-algorithm.
        """
        if fea_run == 0:
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
            self.curr_population.extend(
                [PopulationMember(c, self.calc_fitness(c, self.global_solution, self.factor)) for c in children])
            self.select_new_generation(i)
            '''
            Only run this part of the algorithm in the single-population case.
            I.e. when NSGA2 is NOT used as the base-algorithm for FEA.
            '''
            if self.factor is None and i != 1:
                archive_nondom_indeces = find_non_dominated(
                    np.array([np.array(x.fitness) for x in self.nondom_archive]))
                '''
                Update the non-dominated archive.
                '''
                seen = set()
                nondom_archive = [self.nondom_archive[i] for i in archive_nondom_indeces]
                self.nondom_archive = []
                for s in nondom_archive:
                    if s.fitness not in seen:
                        seen.add(s.fitness)
                        self.nondom_archive.append(s)
                if len(self.nondom_archive) == old_archive_length and len(self.nondom_archive) >= 10:
                    change_in_nondom_size.append(True)
                else:
                    change_in_nondom_size = []
                old_archive_length = len(self.nondom_archive)

                '''
                Calculate generation statistics.
                '''
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
                 combinatorial_values=[], factor=None, global_solution=None, archive_size=200):
        super().__init__(combinatorial_values, population_size, dimensions)
        self.dimensions = dimensions
        self.population_size = population_size
        self.ea = evolutionary_algorithm(dimensions, population_size, tournament_size=2, offspring_size=population_size)
        self.curr_population, self.initial_solution = self.initialize_population(gs=global_solution, factor=factor)
        self.factor = factor
        self.global_solution = global_solution
        self.ea_runs = ea_runs
        self.nondom_pop = []
        self.archive_size = archive_size
        self.nondom_archive = []
        self.iteration_stats = []
        self.distance_matrix = []
        self.worst_index = None
    
    def replace_worst_solution(self, gs):
        """

        @param gs: List of Global Solution(s).
                Used to replace the worst solution(s) with.
        Finds the worst solution based on fitness values and replaces with global solution.
        There can be multiple "worst" solutions, in which case as many are replaces as there are global solutions.
        """
        if self.worst_index is None or self.worst_index == []:
            pass
    
    def run(self, fea_run=0):
        """
        Run the full algorithm for the set number of 'ea_runs' or until a convergence criterion is met.
        @param fea_run: Which generation the FEA is on, if NSGA2 is being used as the base-algorithm.
        """

        #initialize population
        if fea_run == 0:
            self.curr_population, self.initial_solution = self.initialize_population(gs=self.global_solution,
                                                                                 factor=self.factor)
        i = 1
        change_in_nondom_size = []
        old_archive_length = 0
        '''
        Convergence criterion based on change in non-dominated solution set size
        '''
        while i != self.ea_runs: #and len(change_in_nondom_size) < 10:
            #calculate strength value fitness for entire population
            strength_pop = self.sorting_mechanism(self.curr_population)
            #copy nondom solutions to archive
            self.nondom_pop = [x for x in strength_pop if x.fitness < 1.0]
            #update archive
            self.update_archive(strength_pop)
            #create new generation
            children = self.ea.create_offspring(self.nondom_archive)
            self.curr_population = []
            self.curr_population.append(
                [PopulationMember(c, self.calc_fitness(c, self.global_solution, self.factor)) for c in children])
            i+=1
    
    def update_archive(self, population):
        self.nondom_archive = [x for x in self.nondom_pop]
        if len(self.nondom_pop) < self.archive_size:
            remaining = [x for x in population if x.fitness >= 1.0]
            remaining = sorted(remaining, key='fitness')
            self.nondom_archive.extend(remaining[:(self.archive_size-len(self.nondom_pop))])
        else:
            distance_matrix = self.calculate_distance_matrix([x.fitness for x in self.nondom_pop])                        
            while len(self.nondom_archive) > self.archive_size:
                most_crowded_idx = self.find_most_crowded(distance_matrix)
                self.remove_point_from_dist_mtx(distance_matrix, most_crowded_idx)
                del self.nondom_archive[most_crowded_idx]
    
    def sorting_mechanism(self, population):
        """
        Code based on Platypus implementation:
        https://platypus.readthedocs.io/en/latest/_modules/platypus/algorithms.html
        """
        fitnesses = np.array([np.array(x.fitness) for x in population])

        raw_strength = [0]*len(population)
        final_strength = [0.0]*len(population)
         
        # compute dominance flags
        keys = list(itertools.combinations(range(len(population)), 2))
        flags = list(map(compare_solutions, [population[k[0]] for k in keys], [population[k[1]] for k in keys]))

        self.distance_matrix = self.calculate_distance_matrix(fitnesses)

        # count the number of individuals each solution dominates
        for key, flag in zip(keys, flags):
            if flag < 0:
                raw_strength[key[0]] += 1
            elif flag > 0:
                raw_strength[key[1]] += 1

        # the raw fitness is the sum of the dominance counts (strength) of all
        # dominated solutions
        for key, flag in zip(keys, flags):
            if flag < 0:
                final_strength[key[1]] += raw_strength[key[0]]
            elif flag > 0:
                final_strength[key[0]] += raw_strength[key[1]]

         # add density to fitness
        for i in range(len(population)):
            final_strength[i] += 1.0 / (self.kth_distance(i, self.k) + 2.0)
        
        return [PopulationMember(x.variables, fs) for x, fs in zip(population, final_strength)]


    def calculate_distance_matrix(self, fitnesses, distance_fun=euclidean_distance):
        """
        Code from Platypus library!
        Maintains the pairwise distances between solutions.  
        It also provides convenient routines to lookup the distance between any two solutions, 
        find the most crowded solution, and remove a solution.
        """
        distances = []
        for i in range(len(fitnesses)):
            distances_i = []
            for j in range(len(fitnesses)):
                if i != j:
                    distances_i.append((j, distance_fun(fitnesses[i], fitnesses[j])))
                      
            distances.append(sorted(distances_i, key=lambda x : x[1]))
        return distances
    
    def find_most_crowded(self, distance_matrix):
        """
        Code from Platypus library!
        Finds the most crowded solution.
        
        Returns the index of the most crowded solution, which is the solution
        with the smallest distance to the nearest neighbor.  Any ties are
        broken by looking at the next distant neighbor.
        """
        minimum_distance = np.inf
        minimum_index = -1
        
        for i in range(len(distance_matrix)):
            distances_i = self.distance_matrix[i]
            
            if distances_i[0][1] < minimum_distance:
                minimum_distance = distances_i[0][1]
                minimum_index = i
            elif distances_i[0][1] == minimum_distance:
                for j in range(len(distances_i)):
                    dist1 = distances_i[j][1]
                    dist2 = distance_matrix[minimum_index][j][1]
                    
                    if dist1 < dist2:
                        minimum_index = i
                        break
                    if dist2 < dist1:
                        break
        
        return minimum_index
    
    def remove_point_from_dist_mtx(self, distance_matrix, index):
        """
        Code from Platypus library!
        Removes the distance entries for the given solution.
        
        Parameters
        ----------
        index : int
            The index of the solution
        """
        del distance_matrix[index]
        
        for i in range(len(self.distances)):
            distance_matrix[i] = [(x if x < index else x-1, y) for (x, y) in distance_matrix[i] if x != index]
    
    def kth_distance(self, i, k):
        """
        Code from Platypus library!
        Returns the distance to the k-th nearest neighbor.
        
        Parameters
        ----------
        i : int
            The index of the solution
        k : int
            Finds the k-th nearest neightbor distance
        """
        return self.distance_matrix[i][k][1]



class MOEAD(MOOEA):
    """

    """
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
