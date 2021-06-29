"""
The Genetic Algorithm implementation to maximize stratification while minimizing jumps between consecutive cells.

Prescription class -- stores all relevant fitness score information for each map instance.
GA class -- algorithm instance
"""

import numpy as np
import random, datetime, csv, os, operator
from copy import deepcopy, copy
from refactoring.optimizationproblems.prescription import Prescription


class GA:
    def __init__(self, function, population_size=200, tournament_size=20, mutation_rate=0.1, crossover_rate=0.90,
                 ga_runs=20, mutation_type="swap", crossover_type="multi",
                 parent_pairs_size=20, data_distribution=False, weight=.75, factor=None):
        self.function = function
        self.run_algorithm_bool = False
        self.prescription_maps = []
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.field = None
        self.factor = factor
        self.ga_runs = ga_runs
        self.mutation_type = mutation_type
        self.crossover_type = crossover_type
        self.parent_pairs_size = parent_pairs_size
        self.cell_distribution = data_distribution  # bins are equally distributed across cells: True or False
        self.stopping_run = False
        self.w = weight
        self.expected_nitrogen_strat = 0
        self.max_strat = 0
        self.min_strat = 0

    def stop_ga_running(self):
        self.stopping_run = True

    def set_field(self, field):
        self.field = field

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
        keys = ["Run", "Fitness_score", "Jumps_score", "Stratificiation_score", "Variance", "Min_score", "Max_score"]
        stat_values = []
        scores = [_map.score for _map in self.prescription_maps]
        stat_values.append(run)
        stat_values.append(np.mean(scores))
        stat_values.append(np.mean([_map.jumps for _map in self.prescription_maps]))
        stat_values.append(np.mean([_map.strat for _map in self.prescription_maps]))
        stat_values.append(np.var(scores))
        stat_values.append(min(scores))
        stat_values.append(max(scores))
        stats_dictionary = dict(zip(keys, stat_values))
        return stats_dictionary

    def generate_prescriptions(self, index):
        """
        creates a map, along with all the relevant info about jumps and stratification
        """
        prescript_class = Prescription()
        modified_cells = self.field.assign_nitrogen_distribution()
        prescript_class.set_fitness(self.function, modified_cells)
        prescript_class.index = index
        prescript_class.variables = deepcopy(modified_cells)

        return prescript_class

    def tournament_selection(self, k_individuals):
        """
        Picks a number of maps and finds and returns the map dict with the best prescription score
        """
        i = 0
        chosen_prescriptions = []
        while i < k_individuals:
            rand = random.randint(0, len(self.prescription_maps) - 1)
            chosen_prescriptions.append(self.prescription_maps[rand])
            i += 1
        chosen_prescriptions.sort(key=operator.attrgetter('score'))

        return chosen_prescriptions[0]

    def mutate(self, original_map):
        """
        Type = Swap, scramble or inversion
        """
        num_cells = len(original_map.cell_list)
        rand = random.random()
        _map = deepcopy(original_map)
        if self.mutation_type == "swap":
            # for index, gene in enumerate(map_dict.cell_list):
            if rand < self.mutation_rate:
                index_1 = random.choice(list(range(0, num_cells)))
                numbers = list(range(0, num_cells))
                numbers.remove(index_1)
                index_2 = random.choice(numbers)
                _map.cell_list[index_1].nitrogen = original_map.cell_list[index_2].nitrogen
                _map.cell_list[index_2].nitrogen = original_map.cell_list[index_1].nitrogen
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
                temp_list = original_map.cell_list[min_index:max_index]
                np.random.shuffle(temp_list)
                _map.cell_list[min_index:max_index] = temp_list

        _map.overall_fitness, _map.jumps, _map.strat, _map.fertilizer_rate \
            = self.function.problem.calculate_overall_fitness(_map.cell_list)

        return _map

    def crossover(self, first_map, second_map):
        """
        Picks two indices and selects everything between those points.
        Type = single, multi or uniform
        """
        _first_map = deepcopy(first_map)
        _second_map = deepcopy(second_map)
        num_cells = len(first_map.cell_list)
        index_1 = random.randint(0, num_cells - 1)
        index_2 = random.randint(0, num_cells - 1)

        if index_1 > index_2:
            max_index = index_1
            min_index = index_2
        else:
            max_index = index_2
            min_index = index_1
        max_index = max_index + 1
        _first_map.cell_list[min_index:max_index] = second_map.cell_list[min_index:max_index]
        _second_map.cell_list[min_index:max_index] = first_map.cell_list[min_index:max_index]
        _first_map.overall_fitness, _first_map.jumps, _first_map.strat, _first_map.fertilizer_rate \
            = self.function.problem.calculate_overall_fitness(_first_map.cell_list)
        _second_map.overall_fitness, _second_map.jumps, _second_map.strat, _second_map.fertilizer_rate \
            = self.function.problem.calculate_overall_fitness(_second_map.cell_list)

        return _first_map, _second_map

    def possible_replacement(self, prescription_maps, given_map):
        """
        Since two children are made every time, two old maps need to be replaced.
        For the first map, it replaces the worst map in the prescription_maps list.
        However, the second map can't replace the first in the case the first is the worst map.
        So the function uses the index of the first maps position in order to avoid that
        """

        scores = [map.score for map in self.prescription_maps]
        min_index = scores.index(min(scores))
        prescription_maps[min_index] = deepcopy(given_map)
        return min_index, prescription_maps

    def run_genetic_algorithm(self, progressbar=None, writer=None):
        """
        Run the entire genetic algorithm
        """
        self.expected_nitrogen_strat, self.max_strat, self.min_strat = self.function.problem.calc_expected_bin_strat()
        i = 0
        initial_map = None
        while i < self.population_size:
            new_map = self.generate_prescriptions(self.field, i)
            self.prescription_maps.append(new_map)
            if i == 0:
                initial_map = new_map
            i = i + 1
        i = 1

        # the GA runs a specified number of times
        while i < self.ga_runs:  # TODO: ADD CONVERGENCE CRITERIUM
            stats_dict = self.calculate_statistics(i)
            children = []
            if writer is not None:
                writer.writerow(stats_dict)

            if progressbar is not None:
                progressbar.update_progress_bar("Genetic Algorithm run: " + str(i) + ". \n Number of jumps: " + str(
                                             stats_dict['Jumps_score']) + ".", i / self.ga_runs * 100)

            j = 0
            while j < self.parent_pairs_size:
                first_map = self.tournament_selection(self.tournament_size)
                self.prescription_maps.remove(first_map)
                second_map = self.tournament_selection(self.tournament_size)
                self.prescription_maps.append(first_map)

                if random.random() < self.crossover_rate:
                    child1, child2 = self.crossover(first_map, second_map)
                else:
                    child1 = deepcopy(first_map)
                    child2 = deepcopy(second_map)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                children.append(child1)
                children.append(child2)
                j += 1

            total_population = self.prescription_maps + children
            total_population.sort(key=operator.attrgetter('score'))
            self.prescription_maps = deepcopy(total_population[:self.population_size])
            random.shuffle(self.prescription_maps)

            i += 1
            if self.stopping_run:
                progressbar.progress_window.destroy()
                break

        # finds the best index and returns that map
        best_index = self.prescription_maps[0].index
        best_score = self.prescription_maps[0].score
        i = 0
        while i < self.population_size:
            if self.prescription_maps[i].score < best_score:
                best_score = self.prescription_maps[i].score
                best_index = self.prescription_maps[i].index
            i = i + 1
        if writer is not None:
            writer.writerow(self.calculate_statistics(self.ga_runs + 1))
        best_map = self.prescription_maps[best_index]
        return best_map, initial_map
