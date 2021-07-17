"""
The Genetic Algorithm implementation to maximize stratification while minimizing jumps between consecutive cells.

Prescription class -- stores all relevant fitness score information for each solution instance.
GA class -- algorithm instance
"""
from refactoring.optimizationproblems.prescription import Prescription
from refactoring.MOO.paretofront import *

import numpy as np
from pymoo.algorithms.nsga2 import calc_crowding_distance
from pymoo.util.nds.non_dominated_sorting import find_non_dominated
from pymoo.util.nds.fast_non_dominated_sort import fast_non_dominated_sort

import random


class NSGA2:
    def __init__(self, population_size=200, tournament_size=5, mutation_rate=0.1, crossover_rate=0.90,
                 ga_runs=100, mutation_type="swap", crossover_type="multi",
                 parent_pairs_size=20, data_distribution=False, weight=.75, factor=None, global_solution=None):
        self.run_algorithm_bool = False
        self.initial_solution = []
        self.curr_population = []
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.factor = factor
        self.po = ParetoOptimization()
        self.global_solution = global_solution
        self.ga_runs = ga_runs
        self.mutation_type = mutation_type
        self.crossover_type = crossover_type
        self.parent_pairs_size = parent_pairs_size
        self.cell_distribution = data_distribution  # bins are equally distributed across cells: True or False
        self.stopping_run = False
        self.field = None
        self.w = weight
        self.gbests = []
        self.nondom_pop = []
        self.nondom_archive = []
        self.iteration_stats = []

    def stop_ga_running(self):
        self.stopping_run = True

    # calculate_statistics of the population of solutions
    def calculate_statistics(self, run):
        """
        Calculate statistics across all prescription solutions for the current run.
        Param:
            run -- generation index
        Returns:
            Dictionary containing statistics:
                overall fitness, jump score, stratification score, variance, worst and best score.
        """
        keys = ["run", "overall", "jumps", "strat", "fertilizer", "variance", "min_score", "max_score"]
        stat_values = []
        scores = [_solution.overall_fitness for _solution in self.curr_population]
        stat_values.append(run)
        stat_values.append(np.mean(scores))
        stat_values.append(np.mean([_solution.jumps for _solution in self.curr_population]))
        stat_values.append(np.mean([_solution.strat for _solution in self.curr_population]))
        stat_values.append(np.mean([_solution.fertilizer_rate for _solution in self.curr_population]))
        stat_values.append(np.var(scores))
        stat_values.append(min(scores))
        stat_values.append(max(scores))
        stats_dictionary = dict(zip(keys, stat_values))
        return stats_dictionary

    def initialize_population(self, field=None):
        i = 0
        initial_solution = None
        while i < self.population_size:
            if field is not None:
                new_solution = Prescription(field=field, index=i, gs=self.global_solution, factor=self.factor)
                self.curr_population.append(new_solution)
                if i == 0:
                    initial_solution = new_solution
                i = i + 1
            else:
                print("No field was defined to base prescription generation on")
        return initial_solution

    def tournament_selection(self, k_individuals):
        """
        Picks a number of solutions and finds and returns the solution dict with the best prescription score
        """
        i = 0
        chosen_prescriptions = []
        while i < k_individuals:
            rand = random.randint(0, len(self.curr_population) - 1)
            chosen_prescriptions.append(self.curr_population[rand])
            i += 1
        return chosen_prescriptions[0]

    def mutate(self, original_solution):
        """
        Type = Swap, scramble or inversion
        """
        num_cells = len(original_solution.variables)
        rand = random.random()
        _solution = Prescription([x for x in original_solution.variables], self.field, gs=self.global_solution, factor=self.factor)
        if self.mutation_type == "swap":
            # for index, gene in enumerate(solution_dict.variables):
            if rand < self.mutation_rate:
                index_1 = random.choice(list(range(0, num_cells)))
                numbers = list(range(0, num_cells))
                numbers.remove(index_1)
                index_2 = random.choice(numbers)
                _solution.variables[index_1] = original_solution.variables[index_2]
                _solution.variables[index_2] = original_solution.variables[index_1]
        elif self.mutation_type == "scramble":
            if rand < self.mutation_rate:
                index_1 = random.choice(list(range(0, num_cells)))
                numbers = list(range(0, num_cells))  # numbers to choose next index from
                numbers.remove(index_1)  # remove already chosen index from these numbers
                index_2 = random.choice(numbers)
                if index_1 > index_2:
                    max_index = index_1
                    min_index = index_2
                else:
                    max_index = index_2
                    min_index = index_1
                temp_list = original_solution.variables[min_index:max_index]
                np.random.shuffle(temp_list)
                _solution.variables[min_index:max_index] = temp_list

        return _solution

    def crossover(self, first_solution, second_solution):
        """
        Picks two indices and selects everything between those points.
        Type = single, multi or uniform
        """
        _first_solution = Prescription([x for x in first_solution.variables], self.field, gs=self.global_solution, factor=self.factor)
        _second_solution = Prescription([x for x in second_solution.variables], self.field, gs=self.global_solution, factor=self.factor)
        num_cells = len(first_solution.variables)
        index_1 = random.randint(0, num_cells - 1)
        index_2 = random.randint(0, num_cells - 1)

        if index_1 > index_2:
            max_index = index_1
            min_index = index_2
        else:
            max_index = index_2
            min_index = index_1
        max_index = max_index + 1
        _first_solution.variables[min_index:max_index] = second_solution.variables[min_index:max_index]
        _second_solution.variables[min_index:max_index] = first_solution.variables[min_index:max_index]
        _first_solution.set_fitness(_first_solution.variables, self.global_solution, self.factor)
        _second_solution.set_fitness(_second_solution.variables, self.global_solution, self.factor)

        return _first_solution, _second_solution

    def possible_replacement(self, prescription_solutions, given_solution):
        """
        Since two children are made every time, two old solutions need to be replaced.
        For the first solution, it replaces the worst solution in the prescription_solutions list.
        However, the second solution can't replace the first in the case the first is the worst solution.
        So the function uses the index of the first solutions position in order to avoid that
        """

        scores = [solution.score for solution in self.curr_population]
        min_index = scores.index(min(scores))
        prescription_solutions[min_index] = [x for x in given_solution]
        return min_index, prescription_solutions

    def find_best_solutions(self):
        # finds index of the 'best' solution
        best_overall_index = self.curr_population[0].index
        best_jump_index = self.curr_population[0].index
        best_strat_index = self.curr_population[0].index
        best_rate_index = self.curr_population[0].index
        best_overall_score = self.curr_population[0].overall_fitness
        best_jump_score = self.curr_population[0].jumps
        best_strat_score = self.curr_population[0].strat
        best_rate_score = self.curr_population[0].fertilizer_rate

        i = 0
        while i < self.population_size:
            if self.curr_population[i].overall_fitness < best_overall_score:
                best_overall_score = self.curr_population[i].overall_fitness
                best_overall_index = self.curr_population[i].index
            if self.curr_population[i].jumps < best_jump_score:
                best_jump_score = self.curr_population[i].jumps
                best_jump_index = self.curr_population[i].index
            if self.curr_population[i].strat < best_strat_score:
                best_strat_score = self.curr_population[i].strat
                best_strat_index = self.curr_population[i].index
            if self.curr_population[i].fertilizer_rate < best_rate_score:
                best_rate_score = self.curr_population[i].fertilizer_rate
                best_rate_index = self.curr_population[i].index
            i = i + 1
        best_solutions = [self.curr_population[best_overall_index], self.curr_population[best_jump_index],
                     self.curr_population[best_strat_index], self.curr_population[best_rate_index]]
        return best_solutions

    def replace_worst_solution(self, gs):
        """
        :param gs: global solution
        After FEA finishes competition, the global solution replaces the worst solution in each subpopulation
        """
        nondom_indeces = find_non_dominated(np.array([np.array(x.objective_values) for x in self.curr_population]))
        self.nondom_pop = [self.curr_population[i] for i in nondom_indeces]
        dominated = [x for x in self.curr_population if x not in self.nondom_pop]
        if len(dominated) != 0:
            diverse_sort = self.diversity_sort(dominated)
        else:
            diverse_sort = self.diversity_sort(self.nondom_pop)
        worst = diverse_sort[-1]
        idx = np.where(np.array(self.curr_population) == worst)
        self.curr_population[idx[0][0]] = Prescription(variables=[gs.variables[i] for i in self.factor], field=self.field, gs= gs, factor=self.factor)
        worst_solution_vars = [x for x in self.global_solution.variables]
        for i, x in zip(self.factor, worst.variables):
            worst_solution_vars[i] = x
        return Prescription(worst_solution_vars, field=self.field)

    def create_offspring(self):
        j = 0
        children = []
        while j < self.parent_pairs_size:
            first_solution = self.tournament_selection(self.tournament_size)
            self.curr_population.remove(first_solution)
            second_solution = self.tournament_selection(self.tournament_size)
            self.curr_population.append(first_solution)

            if random.random() < self.crossover_rate:
                child1, child2 = self.crossover(first_solution, second_solution)
            else:
                child1 = Prescription([x for x in first_solution.variables], self.field, gs=self.global_solution, factor=self.factor)
                child2 = Prescription([x for x in second_solution.variables], self.field, gs=self.global_solution, factor=self.factor)
            child1 = self.mutate(child1)
            child1.set_fitness(global_solution=self.global_solution, factor=self.factor)
            child2 = self.mutate(child2)
            child2.set_fitness(global_solution=self.global_solution, factor=self.factor)

            children.append(child1)
            children.append(child2)
            j += 1
        return children

    def select_new_generation(self, total_population, method='NSGA2'):
        # check for non-domination, then sort based on crowding distance
        fitnesses = np.array([np.array(x.objective_values) for x in total_population])
        nondom_indeces = find_non_dominated(fitnesses)
        self.nondom_pop = [total_population[i] for i in nondom_indeces]
        self.nondom_archive.extend(self.nondom_pop)
        if len(self.nondom_pop) == self.population_size:
            self.curr_population = [x for x in self.nondom_pop]
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
        else:
            sorted_population = self.diversity_sort(self.nondom_pop)
            self.curr_population = sorted_population[:self.population_size]
        random.shuffle(self.curr_population)

    def diversity_sort(self, population):
        fitnesses = np.array([np.array(x.objective_values) for x in population])
        distances = calc_crowding_distance(fitnesses)
        return [x for y,x in sorted(zip(distances, population))]

    def run(self, progressbar=None, writer=None, field=None):
        """
        Run the entire genetic algorithm
        """
        self.field = field
        self.initial_solution = self.initialize_population(field)

        # the GA runs a specified number of times
        i = 1
        while i < self.ga_runs and len(self.nondom_archive) < 200:  # TODO: ADD CONVERGENCE CRITERIUM

            # if progressbar is not None:
            #     progressbar.update_progress_bar("Genetic Algorithm run: " + str(i) + ". \n Number of jumps: " + str(
            #                                  self.stats_dict['Jumps_score']) + ".", i / self.ga_runs * 100)

            children = self.create_offspring()
            total_population = [x for x in self.curr_population]
            total_population.extend(children)
            self.select_new_generation(total_population)
            if self.factor is None and i!=1:
                archive_nondom_indeces = find_non_dominated(
                    np.array([np.array(x.objective_values) for x in self.nondom_archive]))
                nondom_archive = [self.nondom_archive[i] for i in archive_nondom_indeces]
                self.nondom_archive = list(set(nondom_archive))
                eval_dict = self.po.evaluate_solution(self.nondom_archive, [1, 1, 1])
                eval_dict['GA_run'] = i
                eval_dict['ND_size'] = len(self.nondom_archive)
                self.iteration_stats.append(eval_dict)
            i += 1

            if self.stopping_run:
                if progressbar is not None:
                    progressbar.progress_window.destroy()
                break

        if writer is not None:
            writer.writerow(self.calculate_statistics(self.ga_runs + 1))

        self.gbests = self.find_best_solutions()
