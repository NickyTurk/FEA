"""
The Genetic Algorithm implementation to maximize stratification while minimizing jumps between consecutive cells.

Prescription class -- stores all relevant fitness score information for each map instance.
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
        self.w = weight
        self.gbests = []
        self.nondom_pop = []
        self.pf = ParetoOptimization()

    def stop_ga_running(self):
        self.stopping_run = True

    # calculate_statistics of the population of maps
    def calculate_statistics(self, run):
        """
        Calculate statistics across all prescription maps for the current run.
        Param:
            run -- generation index
        Returns:
            Dictionary containing statistics:
                overall fitness, jump score, stratification score, variance, worst and best score.
        """
        keys = ["run", "overall", "jumps", "strat", "fertilizer", "variance", "min_score", "max_score"]
        stat_values = []
        scores = [_map.overall_fitness for _map in self.curr_population]
        stat_values.append(run)
        stat_values.append(np.mean(scores))
        stat_values.append(np.mean([_map.jumps for _map in self.curr_population]))
        stat_values.append(np.mean([_map.strat for _map in self.curr_population]))
        stat_values.append(np.mean([_map.fertilizer_rate for _map in self.curr_population]))
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
                new_map = Prescription(field=field, index=i)
                self.curr_population.append(new_map)
                if i == 0:
                    initial_solution = new_map
                i = i + 1
            else:
                print("No field was defined to base prescription generation on")
        return initial_solution

    def tournament_selection(self, k_individuals):
        """
        Picks a number of maps and finds and returns the map dict with the best prescription score
        """
        i = 0
        chosen_prescriptions = []
        while i < k_individuals:
            rand = random.randint(0, len(self.curr_population) - 1)
            chosen_prescriptions.append(self.curr_population[rand])
            i += 1

        chosen_prescriptions.sort()
        return chosen_prescriptions[0]

    def mutate(self, original_map):
        """
        Type = Swap, scramble or inversion
        """
        num_cells = len(original_map.variables)
        rand = random.random()
        _map = Prescription(original_map.variables)
        if self.mutation_type == "swap":
            # for index, gene in enumerate(map_dict.variables):
            if rand < self.mutation_rate:
                index_1 = random.choice(list(range(0, num_cells)))
                numbers = list(range(0, num_cells))
                numbers.remove(index_1)
                index_2 = random.choice(numbers)
                _map.variables[index_1] = original_map.variables[index_2]
                _map.variables[index_2] = original_map.variables[index_1]
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
                temp_list = original_map.variables[min_index:max_index]
                np.random.shuffle(temp_list)
                _map.variables[min_index:max_index] = temp_list

        _map.set_fitness(_map.variables, self.global_solution, self.factor)

        return _map

    def crossover(self, first_map, second_map):
        """
        Picks two indices and selects everything between those points.
        Type = single, multi or uniform
        """
        _first_map = Prescription(first_map.variables)
        _second_map = Prescription(second_map.variables)
        num_cells = len(first_map.variables)
        index_1 = random.randint(0, num_cells - 1)
        index_2 = random.randint(0, num_cells - 1)

        if index_1 > index_2:
            max_index = index_1
            min_index = index_2
        else:
            max_index = index_2
            min_index = index_1
        max_index = max_index + 1
        _first_map.variables[min_index:max_index] = second_map.variables[min_index:max_index]
        _second_map.variables[min_index:max_index] = first_map.variables[min_index:max_index]
        _first_map.set_fitness(_first_map.variables, self.global_solution, self.factor)
        _second_map.set_fitness(_second_map.variables, self.global_solution, self.factor)

        return _first_map, _second_map

    def possible_replacement(self, prescription_maps, given_map):
        """
        Since two children are made every time, two old maps need to be replaced.
        For the first map, it replaces the worst map in the prescription_maps list.
        However, the second map can't replace the first in the case the first is the worst map.
        So the function uses the index of the first maps position in order to avoid that
        """

        scores = [map.score for map in self.curr_population]
        min_index = scores.index(min(scores))
        prescription_maps[min_index] = [x for x in given_map]
        return min_index, prescription_maps

    def find_best_solutions(self):
        # finds index of the 'best' solution
        best_overall_index, best_jump_index, best_strat_index, best_rate_index = self.curr_population[0].index
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
        best_maps = [self.curr_population[best_overall_index], self.curr_population[best_jump_index],
                     self.curr_population[best_strat_index], self.curr_population[best_rate_index]]
        return best_maps

    def create_offspring(self):
        j = 0
        children = []
        while j < self.parent_pairs_size:
            first_map = self.tournament_selection(self.tournament_size)
            self.curr_population.remove(first_map)
            second_map = self.tournament_selection(self.tournament_size)
            self.curr_population.append(first_map)

            if random.random() < self.crossover_rate:
                child1, child2 = self.crossover(first_map, second_map)
            else:
                child1 = [x for x in first_map]
                child2 = [x for x in second_map]
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)

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
            total_population = [x for x in total_population if x not in self.nondom_pop]
            sorted_population = self.diversity_sort(total_population)
            new_population.extend(sorted_population[:(self.population_size-len(self.nondom_pop))])
            self.curr_population = new_population
        else:
            sorted_population = self.diversity_sort(self.nondom_pop)
            self.curr_population = sorted_population[:self.population_size]
        random.shuffle(self.curr_population)

    def diversity_sort(self, population):
        fitnesses = np.array([x.objective_values for x in population])
        distances = calc_crowding_distance(fitnesses)
        return [x for y,x in sorted(zip(distances, population))]

    def run(self, progressbar=None, writer=None, field=None):
        """
        Run the entire genetic algorithm
        """
        initial_map = self.initialize_population(field)

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
            total_population = self.curr_population + children
            self.select_new_generation(total_population)
            i += 1

            if self.stopping_run:
                if progressbar is not None:
                    progressbar.progress_window.destroy()
                break

        if writer is not None:
            writer.writerow(self.calculate_statistics(self.ga_runs + 1))

        best_maps = self.find_best_solutions()
        self.gbests = best_maps
        return best_maps, initial_map
