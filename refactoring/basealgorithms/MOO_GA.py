"""
The Genetic Algorithm implementation to maximize stratification while minimizing jumps between consecutive cells.

Prescription class -- stores all relevant fitness score information for each solution instance.
GA class -- algorithm instance
"""
from refactoring.optimizationproblems.prescription import Prescription
from refactoring.MOO.paretofront import *

import numpy as np
from pymoo.algorithms.nsga2 import calc_crowding_distance
import random


class GA:
    def __init__(self, population_size=200, tournament_size=20, mutation_rate=0.1, crossover_rate=0.90,
                 ga_runs=20, mutation_type="swap", crossover_type="multi",
                 parent_pairs_size=20, data_distribution=False, weight=.75, factor=None, global_solution=None):
        self.run_algorithm_bool = False
        self.curr_population = []
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.factor = factor
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
        self.pf = ParetoOptimization()

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

        chosen_prescriptions.sort()
        return chosen_prescriptions[0]

    def mutate(self, original_solution):
        """
        Type = Swap, scramble or inversion
        """
        num_cells = len(original_solution.variables)
        rand = random.random()
        _solution = Prescription(original_solution.variables, self.field, gs=self.global_solution, factor=self.factor)
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
        _first_solution = Prescription(first_solution.variables, self.field, gs=self.global_solution, factor=self.factor)
        _second_solution = Prescription(second_solution.variables, self.field, gs=self.global_solution, factor=self.factor)
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
        self.nondom_pop = self.pf.evaluate_pareto_dominance(self.curr_population)
        dominated = [x for x in self.curr_population if x not in self.nondom_pop]
        diverse_sort = self.diversity_sort(dominated)
        worst = diverse_sort[-1]
        idx = np.where(np.array(self.curr_population) == worst)
        self.curr_population[idx[0][0]] = Prescription(variables=[gs.variables[i] for i in self.factor], field=self.field, gs= gs, factor=self.factor)
        return worst

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
                child1 = Prescription(first_solution.variables, self.field, gs=self.global_solution, factor=self.factor)
                child2 = Prescription(second_solution.variables, self.field, gs=self.global_solution, factor=self.factor)
            child1 = self.mutate(child1)
            child1.set_fitness(global_solution=self.global_solution, factor=self.factor)
            child2 = self.mutate(child2)
            child2.set_fitness(global_solution=self.global_solution, factor=self.factor)

            children.append(child1)
            children.append(child2)
            j += 1
        return children

    def select_new_generation(self, total_population):
        # check for non-domination, then sort based on crowding distance
        self.nondom_pop = self.pf.evaluate_pareto_dominance(total_population)
        if len(self.nondom_pop) == self.population_size:
            self.curr_population = [x for x in self.nondom_pop]
        if len(self.nondom_pop) < self.population_size:
            new_population = [x for x in self.nondom_pop]
            for x in new_population:
                total_population.remove(x)
            sorted_population = self.diversity_sort(total_population)
            new_population.extend(sorted_population[:(self.population_size-len(self.nondom_pop))])
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
        initial_solution = self.initialize_population(field)

        # the GA runs a specified number of times
        i = 1
        while i < self.ga_runs:  # TODO: ADD CONVERGENCE CRITERIUM
            stats_dict = self.calculate_statistics(i)

            if writer is not None:
                writer.writerow(stats_dict)

            if progressbar is not None:
                progressbar.update_progress_bar("Genetic Algorithm run: " + str(i) + ". \n Number of jumps: " + str(
                                             stats_dict['Jumps_score']) + ".", i / self.ga_runs * 100)

            children = self.create_offspring()
            total_population = [x for x in self.curr_population]
            total_population.extend(children)
            self.select_new_generation(total_population)
            i += 1

            if self.stopping_run:
                if progressbar is not None:
                    progressbar.progress_window.destroy()
                break

        if writer is not None:
            writer.writerow(self.calculate_statistics(self.ga_runs + 1))

        best_solutions = self.find_best_solutions()
        self.gbests = best_solutions
        return best_solutions, initial_solution
